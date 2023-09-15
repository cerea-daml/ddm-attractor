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
from typing import Callable, Tuple, Any

# External modules
import torch

# Internal modules
from .etkf import ETKFAssimilation

main_logger = logging.getLogger(__name__)


class ETKFGenAssimilation(ETKFAssimilation):
    def __init__(
            self,
            sampler: Any,
            obs_op: Callable,
            obs_std: float = 2,
            inf_factor: float = 1.0
    ):
        super().__init__(
            obs_op=obs_op,
            obs_std=obs_std,
            inf_factor=inf_factor
        )
        self.sampler = sampler

    def _augment_ens(
            self,
            in_tensor: torch.Tensor
    ):
        sampled_ens = self.sampler.sample(
            in_tensor.reshape(-1, 1, in_tensor.size(-1))
        )
        sampled_ens = sampled_ens.reshape(
            *in_tensor.shape[:-2], -1,
            in_tensor.size(-1)
        )
        augmented_ens = torch.concat((in_tensor, sampled_ens), dim=-2)
        return augmented_ens

    @staticmethod
    def _reconstruct_dynamical(
            in_tensor: torch.Tensor,
            n_ens: int = 3
    ):
        ana_perts = in_tensor[..., :n_ens, :]
        ana_perts = ana_perts-ana_perts.mean(dim=-2, keepdims=True)
        return ana_perts + in_tensor.mean(dim=-2, keepdims=True)

    def assimilate(
            self,
            in_tensor: torch.Tensor,
            in_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        augmented_ens = self._augment_ens(in_tensor)
        ana_out, ana_ens = super().assimilate(augmented_ens, in_obs)
        ana_out = self._reconstruct_dynamical(ana_out, in_tensor.size(-2))
        return ana_out, ana_ens
