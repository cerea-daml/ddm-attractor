#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05/06/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Callable, Tuple

# External modules
import torch

# Internal modules
from .etkf_core import estimate_weights

main_logger = logging.getLogger(__name__)


class ETKFAssimilation(object):
    def __init__(
            self,
            obs_op: Callable,
            obs_std: float = 2,
            inf_factor: float = 1.0
    ):
        self.obs_op = obs_op
        self.obs_std = obs_std
        self.inf_factor = inf_factor

    def assimilate(
            self,
            in_tensor: torch.Tensor,
            in_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_tensor_obs = self.obs_op(in_tensor)

        # Estimate weights
        ens_obs_mean = in_tensor_obs.mean(dim=-2, keepdims=True)
        normed_perts = (in_tensor_obs - ens_obs_mean) / self.obs_std
        normed_obs = (in_obs - ens_obs_mean) / self.obs_std
        weights = estimate_weights(normed_perts, normed_obs, self.inf_factor)

        # Estimate posterior
        ens_mean = in_tensor.mean(dim=-2, keepdims=True)
        ens_perts = in_tensor - ens_mean
        out_ens = torch.einsum(
            "...nm,...nk->...mk", weights, ens_perts
        ) + ens_mean
        return out_ens, out_ens
