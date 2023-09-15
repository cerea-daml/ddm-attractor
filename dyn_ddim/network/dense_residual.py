#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.10.22
#
# Created for ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch.nn

# Internal modules
from ..layers import SinusoidalTimeEmbedding, FilmLayer


logger = logging.getLogger(__name__)


class DenseResidualLayer(torch.nn.Module):
    def __init__(
            self,
            n_time_embedding: int = 128,
            n_neurons: int = 64,
            mult: int = 1,
    ) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_neurons, elementwise_affine=False)
        self.film = FilmLayer(n_neurons, n_time_embedding)
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
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        norm_tensor = self.norm(in_tensor)
        conditioned_tensor = self.film(norm_tensor, time_tensor)
        branch_tensor = self.branch(conditioned_tensor) * self.gamma
        out_tensor = in_tensor + branch_tensor
        return out_tensor


class DenseResidualNetwork(torch.nn.Module):
    def __init__(
            self,
            n_inout: int = 3,
            n_time_embedding: int = 128,
            n_hidden: int = 64,
            n_layers: int = 2,
            mult: int = 1
    ) -> None:
        super().__init__()
        self.embedding_layer = SinusoidalTimeEmbedding(dim=n_time_embedding)
        self.input_projection = torch.nn.Linear(n_inout, n_hidden)
        residual_layers = []
        for _ in range(n_layers):
            residual_layers.append(
                DenseResidualLayer(
                    n_time_embedding=n_time_embedding,
                    n_neurons=n_hidden,
                    mult=mult
                )
            )
        self.residual_layers = torch.nn.ModuleList(residual_layers)
        self.output_head = torch.nn.Linear(n_hidden, n_inout)

    def extract_features(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.embedding_layer(time_tensor)
        curr_tensor = self.input_projection(in_tensor)
        for layer in self.residual_layers:
            curr_tensor = layer(curr_tensor, time_embedding)
        return curr_tensor

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        extracted_features = self.extract_features(
            in_tensor=in_tensor, time_tensor=time_tensor
        )
        output_tensor = self.output_head(extracted_features)
        return output_tensor
