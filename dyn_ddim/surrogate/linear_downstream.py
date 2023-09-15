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
from typing import Any, Tuple

# External modules
import torch.nn

# Internal modules
from .downstream import DownstreamModule
from .least_squares import LeastSquaresModel

main_logger = logging.getLogger(__name__)


class LinearRegressionModule(DownstreamModule):
    def __init__(
            self,
            backbone: torch.nn.Module,
            concat_state: bool = False,
            l2_reg: float = 1E-4,
    ):
        DownstreamModule.__init__(
            self=self,
            backbone=backbone,
            concat_state=concat_state,
        )
        examples_features = self.backbone(self.example_input_array)
        n_features = examples_features.size(-1)
        if self.concat_state:
            n_features += 3
        self.head = LeastSquaresModel(
            n_features=n_features, n_out=3, lam_l2=l2_reg
        )
        self.automatic_optimization = False

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(in_tensor)
        return self.head(features)

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> Any:
        loss = self._get_loss(batch)
        features = self.extract_features(batch[0])
        self.head.record_batch(features, batch[1])
        self.head.estimate_coeffs()
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += 1
        return loss
