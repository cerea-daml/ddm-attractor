#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 02/05/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
import os
from typing import Any

# External modules
import pytorch_lightning as pl

# Internal modules

main_logger = logging.getLogger(__name__)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    Copied from: https://github.com/Lightning-AI/lightning/issues/2534#issuecomment-674582085
    """

    def __init__(
        self,
        save_step_frequency: int = 1,
        prefix="step_checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.internal_step = 0

    def on_train_batch_end(
            self, trainer: pl.Trainer, *args: Any, **kwargs: Any
    ) -> None:
        """ Check if we should save a checkpoint after every train batch """
        self.internal_step += 1
        step = self.internal_step
        if self.internal_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{step=}.ckpt"
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename
            )
            trainer.save_checkpoint(ckpt_path, weights_only=True)
