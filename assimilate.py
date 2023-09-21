#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05/06/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from collections.abc import MutableMapping

# External modules
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate

import wandb

from omegaconf import DictConfig
from tqdm import tqdm

# Internal modules
from dyn_ddim.dynamical import RK4Integrator, Lorenz63
import dyn_ddim.climatology as clim


main_logger = logging.getLogger(__name__)


def flatten_dict(dictionary, parent_key='', separator='.'):
    # Based on https://stackoverflow.com/a/6027615
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(
                flatten_dict(value, new_key, separator=separator).items()
            )
        else:
            items.append((new_key, value))
    return dict(items)


@hydra.main(
    version_base=None, config_path='configs/', config_name='assimilate'
)
def main_assimilate(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load the dataset
    truth = torch.load(cfg.dataset_path)
    clim_scaling = torch.tensor(clim.scale)
    truth = truth * clim_scaling + torch.tensor(clim.shift)
    main_logger.info("Loaded the truth")

    # Generate the observations
    obs_op = lambda state: state[..., cfg.obs_list]
    obs = obs_op(truth)
    obs = obs + torch.randn_like(obs) * cfg.obs_std
    main_logger.info("Generated the observations")

    # Generated initial ensemble
    n_ens_initial = truth.size(0) * cfg.n_ens
    state_clim = truth.view(-1, 3)
    ens_initial = state_clim[
        torch.randperm(state_clim.size(0))[:n_ens_initial]
    ]
    ens_initial = ens_initial.reshape(truth.size(0), cfg.n_ens, 3)
    main_logger.info("Generate initial ensemble")

    assimilation = instantiate(
        cfg.assimilation, obs_std=cfg.obs_std, obs_op=obs_op
    )
    main_logger.info(f"Instantiated {cfg.assimilation._target_}")

    model = Lorenz63()
    integrator = RK4Integrator(model, dt=0.01)
    main_logger.info("Instantiated the model")

    # Instantiate logger
    _ = instantiate(cfg.logger)
    wandb.config["assimilation"] = flatten_dict(cfg.assimilation)
    wandb.config["obs_std"] = cfg.obs_std
    wandb.config["obs_every"] = cfg.obs_every
    wandb.config["obs_list"] = cfg.obs_list

    # Burn in phase
    curr_state = ens_initial.clone()
    mse = 0
    spread = 0
    n_stat_steps = 0

    time_pbar = tqdm(range(1, (cfg.obs_every*cfg.n_burn_in)+1))
    for burn_time in time_pbar:
        curr_state = integrator.integrate(curr_state)
        if (burn_time % cfg.obs_every) == 0:
            # Assimilate
            curr_state, _, curr_ens = assimilation.assimilate(
                curr_state, obs[:, [burn_time]]
            )
            # Estimate statistics
            old_stat_steps = n_stat_steps
            n_stat_steps = n_stat_steps + 1
            old_weight = old_stat_steps / n_stat_steps

            # Update MSE and spread
            curr_mse = (
                    curr_state.mean(dim=-2)-truth[:, burn_time]
            ).pow(2).mean(dim=0)
            mse = mse * old_weight + curr_mse / n_stat_steps
            curr_spread = curr_ens.var(dim=-2).mean(dim=0)
            spread = spread * old_weight + curr_spread / n_stat_steps

            # Estimate local statistics
            curr_nrmse = (curr_mse/clim_scaling.pow(2)).mean().sqrt().item()
            curr_nspread = (
                    curr_spread/clim_scaling.pow(2)
            ).mean().sqrt().item()
            ana_nrmse = (mse/clim_scaling.pow(2)).mean().sqrt().item()
            ana_nspread = (spread/clim_scaling.pow(2)).mean().sqrt().item()

            wandb.log({
                "assim/curr_rmse": curr_nrmse,
                "assim/curr_spread": curr_nspread,
                "assim/ana_rmse": ana_nrmse,
                "assim/ana_spread": ana_nspread
            },)

            # Update rolling statistics
            time_pbar.set_postfix(
                curr_rmse=curr_nrmse, curr_spread=curr_nspread,
                ana_rmse=ana_nrmse, ana_spread=ana_nspread
            )

    main_logger.info("Burn-in phase finished")

    # Estimate statistics
    total_steps = cfg.obs_every * cfg.n_cycles
    n_stat_steps = 0
    mse = torch.zeros(curr_state.size(0), cfg.obs_every+1, 3)
    spread = torch.zeros(curr_state.size(0), cfg.obs_every+1, 3)
    ana_mse = 0
    ana_spread = 0
    bg_mse = 0
    bg_spread = 0
    cov_ana = torch.zeros(3, 3)
    cov_bg = torch.zeros(3, 3)
    curr_traj = [curr_state.clone()]
    time_pbar = tqdm(range(burn_time+1, burn_time+total_steps+1))
    for t in time_pbar:
        curr_traj.append(integrator.integrate(curr_traj[-1]))
        if (t % cfg.obs_every) == 0:
            # Estimate statistics
            old_stat_steps = n_stat_steps
            n_stat_steps = n_stat_steps + 1
            old_weight = old_stat_steps / n_stat_steps

            # Concatenate trajectory
            curr_traj = torch.stack(curr_traj, dim=1)

            # Update MSE and spread
            curr_mse = (
                    curr_traj.mean(dim=-2)-truth[:, t-cfg.obs_every:t+1]
            ).pow(2)
            mse = mse * old_weight + curr_mse / n_stat_steps

            if cfg.n_ens > 1:
                spread = spread * old_weight \
                         + curr_traj.var(dim=-2) / n_stat_steps

                # Update bg cov
                bg_mean = curr_traj[:, -1].mean(dim=-2, keepdims=True)
                bg_perts = curr_traj[:, -1]-bg_mean
                curr_cov = torch.einsum("big,bih->bgh", bg_perts, bg_perts)
                curr_cov = curr_cov.mean(dim=0) / (bg_perts.size(-2)-1)
                cov_bg = cov_bg * old_weight + curr_cov / n_stat_steps

            # Assimilate
            analysis, bg_ens, ana_ens = assimilation.assimilate(
                curr_traj[:, -1], obs[:, [t]]
            )

            # Update background scores
            curr_bg_mse = (bg_ens.mean(dim=-2)-truth[:, t]).pow(2).mean(dim=0)
            bg_mse = bg_mse * old_weight + curr_bg_mse / n_stat_steps
            curr_bg_spread = bg_ens.var(dim=-2).mean(dim=0)
            bg_spread = bg_spread * old_weight + curr_bg_spread / n_stat_steps

            # Update analysis scores
            curr_ana_mse = (analysis.mean(dim=-2)-truth[:, t]).pow(2).mean(dim=0)
            ana_mse = ana_mse * old_weight + curr_ana_mse / n_stat_steps
            curr_ana_spread = ana_ens.var(dim=-2).mean(dim=0)
            ana_spread = ana_spread * old_weight + curr_ana_spread / n_stat_steps

            # Update ana cov
            ana_perts = ana_ens-ana_ens.mean(dim=-2, keepdims=True)
            curr_cov = torch.einsum("big,bih->bgh", ana_perts, ana_perts)
            curr_cov = curr_cov.mean(dim=0) / (ana_perts.size(-2)-1)
            cov_ana = cov_ana * old_weight + curr_cov / n_stat_steps

            # Reset curr_traj
            curr_traj = [analysis]

            # Estimate local statistics
            curr_nrmse = (curr_ana_mse/clim_scaling.pow(2)).mean().sqrt().item()
            curr_nspread = (
                    curr_ana_spread/clim_scaling.pow(2)
            ).mean().sqrt().item()
            bg_nrmse = (bg_mse/clim_scaling.pow(2)).mean().sqrt().item()
            bg_nspread = (bg_spread/clim_scaling.pow(2)).mean().sqrt().item()
            ana_nrmse = (ana_mse/clim_scaling.pow(2)).mean().sqrt().item()
            ana_nspread = (ana_spread/clim_scaling.pow(2)).mean().sqrt().item()

            wandb.log({
                "assim/curr_rmse": curr_nrmse,
                "assim/curr_spread": curr_nspread,
                "assim/bg_rmse": bg_nrmse,
                "assim/bg_spread": bg_nspread,
                "assim/ana_rmse": ana_nrmse,
                "assim/ana_spread": ana_nspread,
            },)

            # Update rolling statistics
            time_pbar.set_postfix(
                curr_rmse=curr_nrmse, curr_spread=curr_nspread,
                ana_rmse=ana_nrmse, ana_spread=ana_nspread
            )

    wandb.define_metric("lead_time")
    wandb.define_metric("assim/rmse_mean", step_metric="lead_time")
    wandb.define_metric("assim/rmse_std", step_metric="lead_time")
    wandb.define_metric("assim/spread_mean", step_metric="lead_time")
    rmse = (mse / clim_scaling.pow(2)).mean(dim=-1).sqrt()
    spread = (spread / clim_scaling.pow(2)).mean(dim=-1).sqrt()
    rmse_mean = rmse.mean(dim=0)
    rmse_std = rmse.std(dim=0)
    spread_mean = spread.mean(dim=0)
    all_scores = zip(rmse_mean, rmse_std, spread_mean)
    for ld, scores in enumerate(all_scores):
        score_dict = {
            "assim/rmse_mean": scores[0],
            "assim/rmse_std": scores[1],
            "assim/spread_mean": scores[2],
            "lead_time": ld
        }
        wandb.log(score_dict)

    wandb.run.summary["cov_bg"] = cov_bg
    wandb.run.summary["cov_ana"] = cov_ana
    wandb.finish()

    rmse_norm = (mse[0]/clim_scaling.pow(2)).mean().sqrt()
    main_logger.info("RMSE: {0:.2f}".format(rmse_norm))
    return rmse_norm


if __name__ == "__main__":
    main_assimilate()
