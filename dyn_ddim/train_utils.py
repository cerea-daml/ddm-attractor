#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.02.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import subprocess

# External modules
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# Internal modules


logger = logging.getLogger(__name__)


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).decode('ascii').strip()


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
) -> dict:
    hparams = {}

    hparams["trainer"] = config["trainer"]
    hparams["network"] = config["network"]
    hparams["datamodule"] = config["data"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # Training parameters
    hparams["seed"] = config["seed"]
    hparams["batch_size"] = config["batch_size"]
    hparams["learning_rate"] = config["learning_rate"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return hparams
