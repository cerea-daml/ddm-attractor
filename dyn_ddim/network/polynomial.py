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
from ..layers import SinusoidalTimeEmbedding, PolynomialLayer, FilmLayer

main_logger = logging.getLogger(__name__)


class PolynomialNetwork(torch.nn.Module):
    def __init__(self, degree: int = 2, n_time_embedding: int = 128):
        super().__init__()
        self.poly_layer = PolynomialLayer(degree=degree, bias=False)
        self.embedding_layer = SinusoidalTimeEmbedding(n_time_embedding)
        self.film_layer = FilmLayer(
            self.poly_layer.n_features, n_time_embedding
        )
        self.output_layer = torch.nn.Linear(self.poly_layer.n_features, 3)

    def extract_features(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.embedding_layer(time_tensor)
        features = self.poly_layer(in_tensor)
        features = self.film_layer(features, time_embedding)
        return features

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        features = self.extract_features(in_tensor, time_tensor)
        return self.output_layer(features)
