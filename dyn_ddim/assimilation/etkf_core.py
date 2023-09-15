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
from math import sqrt

# External modules
import torch
import torch.linalg

# Internal modules

main_logger = logging.getLogger(__name__)


def estimate_weights(
            normed_ens: torch.Tensor,
            normed_obs: torch.Tensor,
            inf_factor: float = 1.0
    ) -> torch.Tensor:
        n_ens = normed_ens.size(-2)
        u, s, v = torch.linalg.svd(normed_ens)
        factor = (n_ens - 1) / inf_factor
        inv_mean = torch.diagonal_scatter(
            torch.zeros_like(normed_ens),
            s / (s ** 2 + factor),
            dim1=-2, dim2=-1
        )
        w_mean = torch.einsum(
            "...ij,...jk,...kl,...ml->...im",
            u, inv_mean, v, normed_obs
        )
        inv_cov = torch.ones(*u.shape[:-1]) * factor
        inv_cov[..., :s.size(-1)] += s ** 2
        inv_cov = 1 / inv_cov.sqrt()
        w_perts = torch.einsum("...ij,...j,...lj->...il", u, inv_cov, u)
        w_perts = sqrt(n_ens-1) * w_perts
        return w_mean + w_perts
