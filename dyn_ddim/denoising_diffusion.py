#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 03.10.22
#
# Created for ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Any, Union

# External modules
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn.functional as F

# Internal modules
from .head_param.head import HeadParam
from .scheduler.noise_scheduler import NoiseScheduler


logger = logging.getLogger(__name__)


class DenoisingDiffusionModule(LightningModule):
    def __init__(
            self,
            network: DictConfig,
            head: DictConfig,
            scheduler: DictConfig,
            timesteps: int = 1000,
            lr: float = 1E-3,
            sampler: DictConfig = None
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.scheduler: NoiseScheduler = instantiate(
            scheduler, timesteps=timesteps
        )
        self.head: HeadParam = instantiate(head)
        self.denoising_network: torch.nn.Module = instantiate(network)
        self.lr = lr
        self.sampler = None
        if sampler is not None:
            self.sampler = instantiate(
                sampler, scheduler=scheduler, head=head,
                denoising_model=self.denoising_network
            )
        self.save_hyperparameters()

    def forward(self, in_tensor: torch.Tensor, idx_time: torch.Tensor):
        return self.denoising_network(in_tensor, idx_time)

    def sample_time(
            self,
            template_tensor: torch.Tensor
    ) -> torch.Tensor:
        time_shape = torch.Size(
            [template_tensor.shape[0]] + [1, ] * (template_tensor.ndim-1)
        )
        sampled_time = torch.randint(
            1, self.timesteps+1, time_shape,
            device=template_tensor.device
        ).long()
        return sampled_time

    def diffuse(
            self,
            in_data: torch.Tensor,
            alpha_sqrt: Any,
            sigma: Any,
    ) -> [torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(in_data)
        noised_data = alpha_sqrt * in_data + sigma * noise
        return noised_data, noise

    def estimate_loss(
            self,
            batch: torch.Tensor,
            prefix: str = "train"
    ) -> torch.Tensor:
        sampled_time = self.sample_time(batch)
        alpha_sqrt = self.scheduler.get_alpha(sampled_time).sqrt()
        sigma = self.scheduler.get_sigma(sampled_time)
        noised_data, noise = self.diffuse(batch, alpha_sqrt, sigma)
        prediction = self.denoising_network(noised_data, sampled_time)
        noise_loss = self.head(
            state=batch,
            noise=noise,
            prediction=prediction,
            alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        self.log(f'{prefix}/loss', noise_loss, on_step=False, on_epoch=True,
                 prog_bar=True)
        denoised_batch = self.head.get_state(
            latent_state=noised_data,
            prediction=prediction,
            alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        data_loss = F.l1_loss(denoised_batch, batch)
        self.log(f'{prefix}/data_loss', data_loss, on_step=False, on_epoch=True,
                 prog_bar=True)
        return noise_loss
    
    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='train')
        return total_loss

    def validation_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='val')
        return total_loss

    def test_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='test')
        return total_loss

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        if self.sampler is None:
            raise ValueError("To predict with diffusion model, "
                             "please set sampler!")
        return self.sampler.sample(batch.shape)

    def configure_optimizers(
            self
    ) -> "torch.optim.Optimizer":
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        return optimizer
