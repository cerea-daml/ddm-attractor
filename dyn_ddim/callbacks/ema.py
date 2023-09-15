#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14.11.22
#
# Created for ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from copy import deepcopy
from typing import Optional, Union, List, Any, Dict

# External modules
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import torch

# Internal modules


logger = logging.getLogger(__name__)


class EMA(pl.Callback):
    """Exponential moving average
    """
    def __init__(
            self,
            decay: float = 0.9999,
            use_ema_weights: bool = True,
            ema_device: Optional[Union[torch.device, str]] = None,
            pin_memory=True
    ):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False
        self._use_ema_weights = use_ema_weights
        self._ema_params: List[torch.Tensor] = []
        self._buffer = None
        self._ema_ready = False

    def _get_network(self, pl_module: pl.LightningModule):
        return pl_module.denoising_network

    def on_train_start(self, trainer: "pl.Trainer",
                       pl_module: pl.LightningModule) -> None:
        if not self._ema_ready and pl_module.global_rank == 0:
            self._ema_params = [
                torch.zeros_like(p) for p
                in self._get_network(pl_module).state_dict().values()
            ]
            if self.ema_device is not None:
                self._ema_params = [
                    tensor.to(device=self.ema_device)
                    for tensor in self._ema_params
                ]
            if self.ema_device == "cpu" and self.ema_pin_memory:
                self._ema_params = [
                    tensor.pin_memory() for tensor in self._ema_params
                ]

        self._ema_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer",
                           pl_module: pl.LightningModule, *args,
                           **kwargs) -> None:
        with torch.no_grad():
            zipped_vals = zip(
                list(self._get_network(pl_module).state_dict().values()),
                self._ema_params
            )
            for orig_weight, ema_weight in zipped_vals:
                ema_weight.copy_(
                    self.decay * ema_weight +
                    (1 - self.decay) * orig_weight.to(ema_weight),
                    non_blocking=True
                )

    def _set_ema_weights(self, pl_module: pl.LightningModule):
        self._buffer = [
            p.detach().clone().to('cpu')
            for p in self._get_network(pl_module).state_dict().values()
        ]
        ema_state_dict = dict(zip(
            self._get_network(pl_module).state_dict().keys(), self._ema_params
        ))
        self._get_network(pl_module).load_state_dict(state_dict=ema_state_dict)

    def _set_original_weights(self, pl_module: pl.LightningModule):
        orig_state_dict = dict(zip(
            self._get_network(pl_module).state_dict().keys(), self._buffer
        ))
        self._get_network(pl_module).load_state_dict(state_dict=orig_state_dict)
        self._buffer = None

    def on_validation_start(self, trainer: pl.Trainer,
                            pl_module: pl.LightningModule) -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_ema_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer",
                          pl_module: "pl.LightningModule") -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_original_weights(pl_module)

    def on_test_start(self, trainer: pl.Trainer,
                            pl_module: pl.LightningModule) -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_ema_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer",
                          pl_module: "pl.LightningModule") -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_original_weights(pl_module)

    def on_predict_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_ema_weights(pl_module)

    def on_predict_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        if self._ema_ready and self._use_ema_weights:
            self._set_original_weights(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": self._ema_params,
            "_ready": self._ema_ready,
            "_use_weights": self._use_ema_weights
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._ema_ready = state_dict.get("_ready")
        self._use_ema_weights = state_dict.get("_use_weights")
        self._ema_params = state_dict.get("params")
