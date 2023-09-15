#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch.nn

# Internal modules
from ..layers import SinusoidalTimeEmbedding, RandomFourierLayer, FilmLayer

main_logger = logging.getLogger(__name__)


class RandomFourierNetwork(torch.nn.Module):
    def __init__(
            self,
            n_neurons: int = 512,
            concat_time: bool = True,
            n_time_embedding: int = 128
    ):
        super().__init__()
        self.concat_time = concat_time
        if self.concat_time:
            self.fourier_layer = RandomFourierLayer(
                in_features=4, n_neurons=n_neurons
            )
        else:
            self.fourier_layer = RandomFourierLayer(
                in_features=3, n_neurons=n_neurons
            )
            self.embedding_layer = SinusoidalTimeEmbedding(n_time_embedding)
            self.film_layer = FilmLayer(
                n_neurons, n_time_embedding
            )
        self.output_layer = torch.nn.Linear(n_neurons, 3)

    def extract_features(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self.concat_time:
            time_tensor = time_tensor / 1000.
            in_tensor = torch.cat((in_tensor, time_tensor), dim=-1)
            out_tensor = self.fourier_layer(in_tensor)
        else:
            out_tensor = self.fourier_layer(in_tensor)
            time_embedding = self.embedding_layer(time_tensor)
            out_tensor = self.film_layer(out_tensor, time_embedding)
        return out_tensor

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        features = self.extract_features(in_tensor, time_tensor)
        return self.output_layer(features)
