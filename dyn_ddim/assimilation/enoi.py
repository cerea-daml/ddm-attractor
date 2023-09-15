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
from typing import Callable, Any, Tuple

# External modules
import torch

# Internal modules
from .etkf import ETKFAssimilation, estimate_weights

main_logger = logging.getLogger(__name__)


class EnOIAssimilation(ETKFAssimilation):
    def __init__(
            self,
            sampler: Any,
            obs_op: Callable,
            obs_std: float = 2.0,
    ) -> None:
        super().__init__(
            obs_op=obs_op,
            obs_std=obs_std
        )
        self.sampler = sampler

    def assimilate(
            self,
            in_tensor: torch.Tensor,
            in_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate ensemble perturbations
        sampled_ens = self.sampler.sample(in_tensor)

        # Estimate weights
        ens_obs = self.obs_op(sampled_ens)
        ens_obs_perts = ens_obs-ens_obs.mean(dim=-2, keepdims=True)
        normed_perts = ens_obs_perts / self.obs_std
        normed_obs = (in_obs - in_tensor) / self.obs_std
        weights = estimate_weights(normed_perts, normed_obs)

        # Estimate posterior
        ens_perts = sampled_ens-sampled_ens.mean(dim=-2, keepdims=True)
        increment = torch.einsum(
            "...nm,...nk->...mk", weights, ens_perts
        )
        out_ens = in_tensor + increment
        out_det = in_tensor + increment.mean(dim=-2, keepdims=True)
        return out_det, out_ens
