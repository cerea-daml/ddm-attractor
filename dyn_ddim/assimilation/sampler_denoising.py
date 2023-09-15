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
from typing import Any

# External modules
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

# Internal modules
from dyn_ddim.denoising_diffusion import DenoisingDiffusionModule
import dyn_ddim.climatology as clim


main_logger = logging.getLogger(__name__)


class DenoisingSampler(object):
    def __init__(
            self,
            ckpt_path: str,
            sampler: DictConfig,
            n_steps: int = 150,
            n_ens: int = 50,
            device: Any = None
    ):
        denoising_model = DenoisingDiffusionModule.load_from_checkpoint(
            ckpt_path, map_location=device
        )
        self.sampler = instantiate(
            sampler,
            scheduler=denoising_model.scheduler,
            head=denoising_model.head,
            denoising_model=denoising_model,
        )
        self.n_steps = n_steps
        self.n_ens = n_ens
        self.clim = torch.tensor(clim.shift), torch.tensor(clim.scale)

    @property
    def device(self) -> torch.device:
        return next(iter(self.sampler.parameters())).device

    def sample(self, in_tensor: torch.Tensor) -> torch.Tensor:
        alpha_t = self.sampler.noise_scheduler.get_alpha(
            int(self.n_steps*self.sampler.step_mul)
        ).sqrt()
        sigma_t = self.sampler.noise_scheduler.get_sigma(
            int(self.n_steps*self.sampler.step_mul)
        )

        normed_det = (in_tensor - self.clim[0]) / self.clim[1]
        normed_det = normed_det.to(self.device)
        scaled_mean = alpha_t * normed_det
        sampled_noise = sigma_t * torch.randn(
            *in_tensor.shape[:-2], self.n_ens, in_tensor.size(-1),
            device=self.device
        )
        latent_ens = scaled_mean + sampled_noise
        with torch.no_grad():
            sampled_ens = self.sampler.reconstruct(latent_ens, self.n_steps)
        sampled_ens = sampled_ens.cpu()
        sampled_ens = sampled_ens * self.clim[1] + self.clim[0]
        return sampled_ens
