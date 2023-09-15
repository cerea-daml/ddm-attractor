#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 17/05/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
import optuna

# Internal modules
from dyn_ddim.surrogate import LinearRegressionModule

main_logger = logging.getLogger(__name__)


class TuneL2Callback(Callback):
    def __init__(
            self,
            lam_limits: Tuple[float, float] = (1E-7, 1E4),
            n_trials: int = 100,
            timeout: float = 120,
            **study_kwargs
    ):
        super().__init__()
        self.lam_limits = lam_limits
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_kwargs = study_kwargs

    def optimize_l2(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> float:
        def objective(trial) -> float:
            lam = trial.suggest_float(
                "lam", self.lam_limits[0], self.lam_limits[1], log=True
            )
            pl_module.head.estimate_coeffs(lam)
            error_sum = torch.zeros(1)
            n_samples = 0
            for in_batch, target_batch in iter(trainer.val_dataloaders):
                in_batch = in_batch.to(pl_module.device)
                with torch.no_grad():
                    prediction = pl_module(in_batch).to('cpu')
                error_sum += (prediction-target_batch).pow(2).sum()
                n_samples += in_batch.size(0)
            return error_sum.item()/n_samples

        study = optuna.create_study(**self.study_kwargs)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        best_lam = study.best_params["lam"]
        return best_lam

    def on_train_epoch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not isinstance(pl_module, LinearRegressionModule):
            raise ValueError(
                "Tuning of the regularization factor works only with a "
                "linear regression module at the moment!"
            )

        main_logger.info(
            "Starting with tuning of L2 regularization parameter"
        )
        best_lam = self.optimize_l2(trainer, pl_module)
        main_logger.info(
            f"Found best L2 regularization parameter: {best_lam}"
        )
        pl_module.head.estimate_coeffs(best_lam)
