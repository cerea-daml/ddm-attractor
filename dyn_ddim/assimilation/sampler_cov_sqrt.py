#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/06/2023
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
from .sampler_cov import CovSampler

main_logger = logging.getLogger(__name__)


class CovSQRTSampler(CovSampler):
    def __init__(
            self,
            covariance: List[List[float]],
            cov_inf: float = 1.
    ):
        super().__init__(
            covariance=covariance,
            n_ens=6,
            cov_inf=cov_inf,
        )
        sigma_points = torch.linalg.cholesky(self.covariance*2.5).T
        sigma_points *= self.cov_inf
        self._perts = torch.cat((sigma_points, -sigma_points), dim=0)

    def sample(self, in_tensor: torch.Tensor) -> torch.Tensor:
        sampled_ens = self._perts + in_tensor
        return sampled_ens
