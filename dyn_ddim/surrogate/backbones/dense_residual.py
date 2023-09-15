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

main_logger = logging.getLogger(__name__)


class DenseResidualLayer(torch.nn.Module):
    def __init__(
            self,
            n_neurons: int = 64,
            mult: int = 2,
    ) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_neurons)
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(n_neurons, n_neurons * mult),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(n_neurons * mult, n_neurons),
        )
        self.gamma = torch.nn.Parameter(
            torch.full((n_neurons, ), fill_value=1E-6), requires_grad=True
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        normed_in = self.norm(in_tensor)
        branch_tensor = self.branch(normed_in) * self.gamma
        out_tensor = in_tensor + branch_tensor
        return out_tensor


class DenseResBackbone(torch.nn.Module):
    def __init__(
            self,
            n_blocks: int = 3,
            hidden_neurons: int = 256,
            mult: int = 1,
    ) -> None:
        super().__init__()
        layers = [torch.nn.Linear(3, hidden_neurons)]
        for _ in range(n_blocks):
            layers.append(DenseResidualLayer(hidden_neurons, mult))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(in_tensor)
