#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.07.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple, Any, Optional
import os

# External modules
import matplotlib
matplotlib.use('agg')

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c

# Internal modules
from ..layers.random_fourier import RandomFourierLayer
from ..sampler.ddpm import DDPMSampler
from dyn_ddim import eval_metrics


logger = logging.getLogger(__name__)


class SampleLorenz63Callback(Callback):
    def __init__(
            self,
            n_samples: int = 256,
            reference_path: str = None,
            plot: bool = False
    ):
        super().__init__()
        self.sampler = None
        self.metrics = {}
        if reference_path is not None:
            reference = torch.load(reference_path).view(-1, 3)
            frechet_network = RandomFourierLayer(n_neurons=512)
            self.metrics = {
                "l2_gen": eval_metrics.NearestCEGen(reference=reference),
                "l2_eval": eval_metrics.NearestCERef(reference=reference),
                "hellinger": eval_metrics.HellingerDistance(
                    reference=reference, bounds=torch.linspace(-3.5, 3.5, 8)
                ),
                "frechet": eval_metrics.FrechetDistance(
                    reference=reference, network=frechet_network,
                    feature_size=512
                )
            }
        self.plot = plot
        self.prior_sample = torch.randn(n_samples, 3)

    def estimate_correlation(
            self,
            sample: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_perts = sample - sample.mean(dim=0)
        sample_cov = sample_perts.T @ sample_perts
        sample_cov /= (sample.shape[0] - 1)
        sample_std = sample_perts.std(dim=0, unbiased=True)
        sample_corr = sample_cov / sample_std[None, :] / sample_std[:, None]
        return sample_cov, sample_corr

    def log_metrics(
            self,
            sample: torch.Tensor,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        for k, metric in self.metrics.items():
            pl_module.log(f"sample/{k:s}", metric(sample))

    def log_figures(
            self,
            sample: torch.Tensor,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        # Estimate covariance / correlation
        sample_cov, sample_corr = self.estimate_correlation(sample)
        # Plot covariance
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.matshow(sample_cov.cpu(), cmap="coolwarm", norm=mpl_c.CenteredNorm())
        trainer.logger.log_image(
            "sample/covariance", [fig]
        )
        # Plot correlation
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.matshow(sample_corr.cpu(), cmap="coolwarm", vmin=-1, vmax=1)
        trainer.logger.log_image(
            "sample/correlation", [fig]
        )
        # Plot attractor
        fig, ax = plt.subplots(ncols=3, figsize=(9, 3))
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
        ax[0].scatter(sample[:, 0].cpu(), sample[:, 1].cpu(), s=2,
                      alpha=0.3, c="firebrick")
        ax[1].scatter(sample[:, 0].cpu(), sample[:, 2].cpu(), s=2,
                      alpha=0.3, c="firebrick")
        ax[2].scatter(sample[:, 1].cpu(), sample[:, 2].cpu(), s=2,
                      alpha=0.3, c="firebrick")

        ax[0].text(x=0.5, y=1.01, s="0/1", ha="center", va="center",
                   transform=ax[0].transAxes)
        ax[1].text(x=0.5, y=1.01, s="0/2", ha="center", va="center",
                   transform=ax[1].transAxes)
        ax[2].text(x=0.5, y=1.01, s="1/2", ha="center", va="center",
                   transform=ax[2].transAxes)
        trainer.logger.log_image(
            "sample/sample", [fig]
        )
        plt.close("all")

    def setup(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            stage: Optional[str] = None
    ) -> None:
        self.sampler = DDPMSampler(
            scheduler=pl_module.scheduler,
            head=pl_module.head,
            timesteps=pl_module.timesteps,
            denoising_model=pl_module.denoising_network
        )

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        if trainer.logger is not None:
            pl_module.eval()
            curr_param = next(pl_module.parameters())
            self.prior_sample = self.prior_sample.to(curr_param)
            sample = self.sampler.reconstruct(
                self.prior_sample, self.sampler.timesteps
            )
            self.log_metrics(sample, trainer, pl_module)
            if self.plot:
                self.log_figures(sample, trainer, pl_module)
