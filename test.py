#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.10.22
#
# Created for ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb

# Internal modules
from dyn_ddim import eval_metrics
from dyn_ddim.callbacks.ema import EMA

main_logger = logging.getLogger(__name__)


def generate_predictions(
        data_module: pl.LightningDataModule,
        trainer: pl.Trainer,
        cfg: DictConfig
) -> torch.Tensor:
    main_logger.info(f"Instantiating network <{cfg.network._target_}>")
    network: pl.LightningModule = instantiate(
        cfg.network, _recursive_=False
    )
    main_logger.info(f"Load network state dict")
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    incompatible_keys = network.load_state_dict(state_dict["state_dict"])
    main_logger.info(f"{incompatible_keys}")

    if cfg.ema:
        try:
            ema_callback = EMA()
            ema_callback.load_state_dict(state_dict["callbacks"]["EMA"])
            ema_callback._set_ema_weights(network)
            main_logger.info(f"Loaded EMA state dict from checkpoint")
        except KeyError:
            main_logger.warn("EMA state dict not found, using without EMA!")

    main_logger.info(f"Instantiating sampler <{cfg.sampler._target_}>")
    sampler = instantiate(
        cfg.sampler, head=network.head, scheduler=network.scheduler,
        denoising_model=network.denoising_network,
    )
    network.sampler = sampler

    main_logger.info(f"Starting prediction")
    predictions = trainer.predict(
        model=network, datamodule=data_module
    )
    predictions = torch.cat(predictions, dim=0)
    return predictions


@hydra.main(version_base=None, config_path='configs/', config_name='test')
def main_test(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)

    metric_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        metric_logger = instantiate(cfg.logger)
        metric_logger.save()
        wandb.config["seed"] = cfg.seed

    main_logger.info(f"Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=None,
        logger=metric_logger
    )

    if cfg.prediction_path is None:
        predictions = generate_predictions(
            data_module=data_module, trainer=trainer, cfg=cfg
        )
    else:
        data_module.setup()
        predictions = torch.load(cfg.prediction_path).view(-1, 3)
    surrogate_model: pl.LightningModule = instantiate(
        cfg.surrogate,
    )
    if OmegaConf.select(cfg, "surrogate_ckpt") is not None:
        surrogate_model.load_state_dict(
            torch.load(cfg.surrogate_ckpt)["state_dict"]
        )
    surrogate_model = surrogate_model.to(trainer.strategy.root_device).eval()
    metrics = {
        "hellinger": eval_metrics.HellingerDistance(
            reference=data_module.test_dataset,
            bounds=torch.linspace(-3.5, 3.5, 71),
        ),
        "frechet": eval_metrics.FrechetDistance(
            reference=data_module.test_dataset,
            network=surrogate_model.backbone,
            batch_size=16384
        ),
        "dist_test": eval_metrics.NearestCERef(
            reference=data_module.test_dataset,
            device=trainer.strategy.root_device
        ),
        "dist_gen": eval_metrics.NearestCEGen(
            reference=data_module.test_dataset,
            device=trainer.strategy.root_device
        ),
        "extreme": eval_metrics.ExtremeValue(
            reference=data_module.test_dataset,
            levels=(0.01, 0.99)
        )
    }
    scores = {
        f"test/{k:s}": metric(predictions)
        for k, metric in metrics.items()
    }
    if metric_logger is not None:
            metric_logger.log_metrics(
                scores, step=0
            )
    main_logger.info(scores)
    wandb.finish()


if __name__ == "__main__":
    main_test()
