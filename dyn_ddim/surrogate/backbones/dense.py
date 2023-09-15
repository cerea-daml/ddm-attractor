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
from typing import Iterable

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class DenseBackbone(torch.nn.Module):
    def __init__(self, hidden_neurons: Iterable[int] = (128, 128),):
        super().__init__()
        layers = [
            torch.nn.Linear(3, hidden_neurons[0]),
            torch.nn.ReLU()
        ]
        for k, curr_neurons in enumerate(hidden_neurons[:-1]):
            layers.extend([
                torch.nn.Linear(curr_neurons, hidden_neurons[k+1]),
                torch.nn.ReLU()
            ])
        self.network = torch.nn.Sequential(*layers)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(in_tensor)
