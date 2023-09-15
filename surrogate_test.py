#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from hydra.utils import instantiate
import wandb

# Internal modules
from dyn_ddim.datasets import TrajDataset

main_logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path='configs/', config_name='surrogate_test'
)
def main_score(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.surrogate)

    dataset = TrajDataset(
        cfg.dataset_path, delta_t=cfg.delta_t,
        length=cfg.n_lead_time
    )
    data_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers
    )

    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        training_logger = instantiate(cfg.logger)
        wandb.config["seed"] = cfg.seed

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=training_logger
    )
    trainer.test(
        model=model, dataloaders=data_loader, ckpt_path=cfg.ckpt_path
    )
    wandb.finish()


if __name__ == "__main__":
    main_score()
