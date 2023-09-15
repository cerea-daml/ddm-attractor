#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24/05/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class LeastSquaresModel(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_out: int,
            lam_l2: float = 1E-4
    ) -> None:
        super().__init__()
        self.register_buffer(
            "running_x_mean", torch.zeros(n_features)
        )
        self.register_buffer(
            "running_x_sq_mean", torch.zeros(n_features)
        )
        self.register_buffer(
            "running_y_mean", torch.zeros(n_out)
        )
        self.register_buffer(
            "seen_samples", torch.zeros(1)
        )
        self.register_buffer(
            "running_xtx", torch.zeros(n_features, n_features)
        )
        self.register_buffer(
            "running_xty", torch.zeros(n_features, n_out)
        )
        self.layer = torch.nn.Linear(n_features, n_out)
        self.lam_l2 = lam_l2
        self.eps = 1E-8

    @property
    def dof(self) -> float:
        bias_correction = self.seen_samples/(self.seen_samples+1)
        xtx = (
            self.running_xtx
            -(self.running_x_mean[:, None] * self.running_x_mean[None, :])
        )/self.x_std[:, None]/self.x_std[None, :]*bias_correction
        eigval = torch.linalg.eigvalsh(xtx)
        return (eigval / (eigval + self.l2_reg)).sum().item()

    @property
    def x_std(self) -> torch.Tensor:
        sum_x_sq = self.running_x_sq_mean * self.seen_samples
        sum_x_mean_sq = self.running_x_mean.pow(2) * self.seen_samples
        x_std = (sum_x_sq-sum_x_mean_sq) / (self.seen_samples+1)
        x_std[x_std < self.eps] = 1.
        x_std = torch.sqrt(x_std)
        return x_std

    @torch.no_grad()
    def record_batch(
            self,
            batch_x: torch.Tensor,
            batch_y: torch.Tensor
    ) -> None:
        new_seen_samples = self.seen_samples + batch_x.size(0)
        self.running_x_mean = (
            self.running_x_mean * self.seen_samples + batch_x.sum(dim=0)
        ) / new_seen_samples
        self.running_x_sq_mean = (
            self.running_x_sq_mean * self.seen_samples
            + batch_x.pow(2).sum(dim=0)
        ) / new_seen_samples
        self.running_y_mean = (
            self.running_y_mean * self.seen_samples + batch_y.sum(dim=0)
        ) / new_seen_samples
        self.running_xtx = (
            self.running_xtx * self.seen_samples + (batch_x.T @ batch_x)
        ) / new_seen_samples
        self.running_xty = (
            self.running_xty * self.seen_samples + (batch_x.T @ batch_y)
        ) / new_seen_samples
        self.seen_samples = new_seen_samples

    def estimate_xtx(self, lam_l2: float) -> torch.Tensor:
        bias_correction = self.seen_samples / (self.seen_samples + 1)
        xtx = (
            self.running_xtx
            - (self.running_x_mean[:, None] * self.running_x_mean[None, :])
        ) / self.x_std[:, None] / self.x_std[None, :] * bias_correction
        reg = torch.eye(
            xtx.size(0), layout=xtx.layout, device=xtx.device, dtype=xtx.dtype
        ) * lam_l2
        xtx = xtx + reg
        return xtx

    def estimate_xty(self) -> torch.Tensor:
        bias_correction = self.seen_samples / (self.seen_samples + 1)
        xty = (
            self.running_xty
            - (self.running_x_mean[:, None] * self.running_y_mean[None, :])
        ) / self.x_std[:, None] * bias_correction
        return xty

    @torch.no_grad()
    def estimate_coeffs(self, lam_l2: float = None) -> None:
        if lam_l2 is None:
            lam_l2 = self.lam_l2
        xtx = self.estimate_xtx(lam_l2)
        xty = self.estimate_xty()
        coeffs = torch.linalg.solve(xtx, xty)
        self.layer.weight.data[:] = coeffs.T
        self.layer.bias.data[:] = self.running_y_mean

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_tensor_norm = (in_tensor-self.running_x_mean) / self.x_std
        return self.layer(in_tensor_norm)
