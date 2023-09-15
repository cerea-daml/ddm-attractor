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
from typing import List

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


class CovSampler(object):
    def __init__(
            self,
            covariance: List[List[float]],
            n_ens: int = 50,
            cov_inf: float = 1.,
    ):
        self.covariance = torch.tensor(covariance)
        self.chol = torch.linalg.cholesky(self.covariance)
        self.cov_inf = cov_inf
        self.n_ens = n_ens

    def sample(self, in_tensor: torch.Tensor) -> torch.Tensor:
        ens_perts = torch.randn(
            *in_tensor.shape[:-2], self.n_ens, in_tensor.size(-1),
            device=in_tensor.device
        )
        ens_perts = ens_perts @ self.chol.to(in_tensor.device).T
        ens_perts -= ens_perts.mean(dim=-2, keepdims=True)
        ens_perts *= self.cov_inf
        sampled_ens = ens_perts + in_tensor
        return sampled_ens
