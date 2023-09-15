#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 02/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging
from math import sqrt
from typing import Tuple

# External modules
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import _compute_fid
from tqdm.autonotebook import tqdm

# Internal modules

main_logger = logging.getLogger(__name__)


class FrechetDistance(object):
    def __init__(
            self,
            reference: torch.Tensor,
            network: torch.nn.Module,
            feature_size: int = 256,
            batch_size: int = 16384,
            num_workers: int = 1
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_size = feature_size

        self.network = network.eval()
        self.reference_statistics = self._estimate_statistics(
            reference
        )

    def _estimate_statistics(self, in_tensor: torch.Tensor):
        loader = DataLoader(
            in_tensor.cpu(), batch_size=self.batch_size, shuffle=False,
            pin_memory=True, num_workers=self.num_workers
        )
        device = next(self.network.parameters()).device
        features_sum = torch.zeros(self.feature_size)
        features_squared_sum = torch.zeros(self.feature_size, self.feature_size)
        n_values = 0
        for sample in tqdm(loader, total=len(loader)):
            sample = sample.to(device)
            with torch.no_grad():
                features = self.network(sample).cpu()
            features = features.view(features.size(0), -1)
            features_sum += features.sum(dim=0)
            features_squared_sum += features.T @ features
            n_values += sample.size(0)
        mean = (features_sum / n_values).unsqueeze(0)
        cov = features_squared_sum - n_values * mean.T.mm(mean)
        cov = cov / (n_values - 1)
        return mean.squeeze(dim=0), cov

    def __call__(
            self,
            sample: torch.Tensor
    ) -> float:
        sample_mean, sample_cov = self._estimate_statistics(sample)
        fid = _compute_fid(
            self.reference_statistics[0].to(torch.float64),
            self.reference_statistics[1].to(torch.float64),
            sample_mean.to(torch.float64),
            sample_cov.to(torch.float64)
        )
        return fid.item()


def hellinger_score(
        reference_hist: torch.Tensor,
        sample_hist: torch.Tensor
) -> torch.Tensor:
    distance = torch.norm(
        reference_hist.sqrt() - sample_hist.sqrt(),
    ).pow(2).sum().sqrt().item()/sqrt(2)
    return distance


class HellingerDistance(object):
    def __init__(
            self,
            reference: torch.Tensor,
            bounds=torch.linspace(-3.5, 3.5, 71)
    ):
        self.reference_hist = torch.histogramdd(
            reference.cpu(), bins=(bounds, bounds, bounds), density=True
        ).hist
        self.reference_hist /= self.reference_hist.sum()
        self.bounds = bounds

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        sample_hist = torch.histogramdd(
            samples.cpu(), bins=(self.bounds, self.bounds, self.bounds),
            density=True
        ).hist
        sample_hist /= sample_hist.sum()
        return hellinger_score(self.reference_hist, sample_hist)


class NearestCEGen(object):
    def __init__(
            self,
            reference: torch.Tensor,
            device: torch.device = torch.device("cpu")
    ):
        try:
            from pykeops.torch import Genred
        except (ImportError or EOFError):
            raise ImportError("Couldn't initialize pykeops")

        formula = "SqDist(x,y)"
        variables = [
            "x = Vi(" + str(reference.shape[-1]) + ")",
            "y = Vj(" + str(reference.shape[-1]) + ")",
        ]
        self.min_func = Genred(
            formula, variables, reduction_op="Min", axis=1
        )
        self.device = device
        self.reference = reference.to(device)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(self.device)
        reference = self.reference.to(self.device)
        with torch.no_grad():
            sq_dist = self.min_func(samples, reference).squeeze().mean()
        return sq_dist.item()


class NearestCERef(NearestCEGen):
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(self.device)
        reference = self.reference.to(self.device)
        with torch.no_grad():
            sq_dist = self.min_func(reference, samples).squeeze().mean()
        return sq_dist.item()


class ExtremeValue(object):
    def __init__(
            self,
            reference: torch.tensor,
            levels: Tuple[float, float] = (0.01, 0.99)
    ):
        self.limits = reference.quantile(torch.tensor(levels), dim=0)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        below_level = (samples <= self.limits[0]).to(torch.float32).mean()
        above_level = (samples >= self.limits[1]).to(torch.float32).mean()
        return (below_level + above_level).item()
