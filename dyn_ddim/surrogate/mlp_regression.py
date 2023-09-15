#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 28/04/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch.nn

# Internal modules
from .downstream import DownstreamModule

main_logger = logging.getLogger(__name__)


class MLPRegressionModule(DownstreamModule):
    def __init__(
            self,
            backbone: torch.nn.Module,
            concat_state: bool = False,
            n_hidden: int = 128,
            lr: float = 1E-3
    ) -> None:
        super().__init__(backbone=backbone, concat_state=concat_state)
        self.lr = lr
        examples_features = self.backbone(self.example_input_array)
        n_features = examples_features.size(-1)
        if self.concat_state:
            n_features += 3
        layers = []
        if n_hidden is not None:
            layers.append(torch.nn.Linear(n_features, n_hidden))
            layers.append(torch.nn.ReLU())
            n_features = n_hidden
        layers.append(torch.nn.Linear(n_features, 3))
        self.head = torch.nn.Sequential(*layers)
        self.save_hyperparameters(ignore="backbone")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
