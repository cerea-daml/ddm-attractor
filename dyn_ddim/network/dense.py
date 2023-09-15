#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 02/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch.nn

# Internal modules
from ..layers.time_embedding import SinusoidalTimeEmbedding
from ..layers.film import FilmLayer

main_logger = logging.getLogger(__name__)


class DenseLayer(torch.nn.Module):
    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            n_time_embedding: int = 128,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_neurons, out_neurons, bias=False)
        self.batch_norm = torch.nn.BatchNorm1d(out_neurons, affine=False)
        self.film = FilmLayer(out_neurons, n_time_embedding)
        self.activation = torch.nn.ReLU()

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        out_tensor = self.linear(in_tensor)
        out_tensor = self.film(out_tensor, time_tensor)
        out_tensor = self.activation(out_tensor)
        return out_tensor


class DenseNetwork(torch.nn.Module):
    def __init__(
            self,
            n_time_embedding: int = 128,
            n_hidden: int = 64,
            n_layers: int = 2,
    ):
        super().__init__()
        self.embedding_layer = SinusoidalTimeEmbedding(dim=n_time_embedding)
        curr_neurons = 3
        layers = []
        for _ in range(n_layers):
            layers.append(
                DenseLayer(
                    curr_neurons, n_hidden, n_time_embedding=n_time_embedding
                )
            )
            curr_neurons = n_hidden
        self.dense_layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(curr_neurons, 3)

    def extract_features(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.embedding_layer(time_tensor)
        curr_tensor = in_tensor
        for layer in self.dense_layers:
            curr_tensor = layer(curr_tensor, time_embedding)
        return curr_tensor

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        features = self.extract_features(in_tensor, time_tensor)
        return self.output_layer(features)
