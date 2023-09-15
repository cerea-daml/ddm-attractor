#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.11.22
#
# Created for ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import PolynomialFeatures

# Internal modules


logger = logging.getLogger(__name__)


def extract_l63_poly_features(
        dataset: Dataset, poly_deg: int = 2, batch_size: int = 4096
) -> torch.Tensor:
    extractor = PolynomialFeatures(degree=poly_deg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    predictor_data = []
    for curr_state in loader:
        curr_predictors = extractor.fit_transform(curr_state.numpy())
        curr_predictors = torch.from_numpy(curr_predictors).to(curr_state)
        predictor_data.append(curr_predictors)
    predictor_data = torch.cat(predictor_data)
    return predictor_data


def extract_features(
        network: torch.nn.Module,
        dataset: Dataset,
        time_tensor: torch.Tensor,
        batch_size: int = 4096,
) -> torch.Tensor:
    use_cuda = next(network.parameters()).is_cuda
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        shuffle=False
    )
    features = []
    for curr_state in loader:
        if use_cuda:
            curr_state = curr_state.cuda()
        with torch.no_grad():
            curr_features = network.extract_features(
                curr_state, time_tensor
            ).cpu()
        features.append(curr_features)
    features = torch.cat(features)
    return features
