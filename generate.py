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
import os.path

# External modules
import hydra
from omegaconf import DictConfig, OmegaConf

main_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs/', config_name='generate')
def main_generate(cfg: DictConfig):
    from hydra.utils import instantiate, get_class
    import pytorch_lightning as pl
    import torch
    from dyn_ddim.callbacks.ema import EMA

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)

    main_logger.info(f"Instantiating network <{cfg.network._target_}>")
    network: pl.LightningModule = instantiate(
        cfg.network, _recursive_=False
    )
    main_logger.info(f"Load network state dict")
    model_checkpoint = instantiate(cfg.callbacks["model_checkpoint"])
    state_dict = torch.load(os.path.join(model_checkpoint.dirpath, "last.ckpt"))
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

    main_logger.info(f"Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=None,
        logger=None
    )

    main_logger.info(f"Starting prediction")
    predictions = trainer.predict(
        model=network, datamodule=data_module
    )
    predictions = torch.cat(predictions, dim=0)

    main_logger.info(f"Store predictions to <{cfg.output_path}>")
    dir_path = os.path.dirname(os.path.abspath(cfg.output_path))
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    torch.save(predictions, cfg.output_path)


if __name__ == "__main__":
    main_generate()