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
import hydra
from omegaconf import DictConfig, OmegaConf

main_logger = logging.getLogger(__name__)


def train_task(cfg: DictConfig) -> float:
    # Import within main loop to speed up training on jean zay
    import wandb
    from hydra.utils import instantiate
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from dyn_ddim.train_utils import log_hyperparameters

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)

    main_logger.info(f"Instantiating network <{cfg.network._target_}>")
    model: pl.LightningModule = instantiate(
        cfg.network,
        lr=cfg.learning_rate,
        _recursive_=False
    )

    if OmegaConf.select(cfg, "callbacks") is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        training_logger = instantiate(cfg.logger)

    if isinstance(training_logger, WandbLogger):
        main_logger.info("Watch gradients and parameters of model")
        hydra_params = log_hyperparameters(config=cfg, model=model)
        training_logger.log_hyperparams(hydra_params)
        training_logger.watch(model, log="all", log_freq=75)

    main_logger.info(f"Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    main_logger.info(f"Starting training")
    trainer.fit(model=model, datamodule=data_module)

    main_logger.info(f"Training finished")
    val_loss = trainer.callback_metrics.get('val/total_loss')
    main_logger.info(f"Validation loss: {val_loss}")
    wandb.finish()
    return val_loss


@hydra.main(version_base=None, config_path='configs/', config_name='config')
def main_train(cfg: DictConfig) -> float:
    import numpy as np
    try:
        val_loss = train_task(cfg)
    except MemoryError:
        val_loss = np.inf
    return val_loss


if __name__ == '__main__':
    main_train()
