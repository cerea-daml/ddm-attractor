#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 28/04/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}
# System modules
import logging
from typing import Any, Tuple

# External modules
import pytorch_lightning as pl
import torch.nn
import wandb

# Internal modules
from dyn_ddim.eval_metrics import hellinger_score

main_logger = logging.getLogger(__name__)


class DownstreamModule(pl.LightningModule):
    def __init__(
            self,
            backbone: torch.nn.Module,
            concat_state: bool = False,
    ):
        super().__init__()
        self.concat_state = concat_state
        self.example_input_array = torch.randn(1, 3)
        self.backbone = backbone
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        self.test_step_scores = []

    def extract_features(self, in_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(in_tensor)
        if self.concat_state:
            features = torch.cat([features, in_tensor], dim=-1)
        return features

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(in_tensor)
        prediction = self.head(features)
        return prediction

    def _get_loss(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            prefix: str = "train"
    ) -> torch.Tensor:
        state_tensor, target_tensor = batch[0], batch[1]
        prediction = self(state_tensor)
        mse = self.mse_loss(prediction, target_tensor)
        mae = self.mae_loss(prediction, target_tensor)
        self.log(f"{prefix:s}/loss", mse, on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log(f"{prefix:s}/mse", mse, on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log(f"{prefix:s}/mae", mae, on_step=False, on_epoch=True,
                 prog_bar=False)
        return mse

    def training_step(
            self,
            batch: Any,
            batch_idx: int
    ) -> Any:
        return self._get_loss(batch)

    def validation_step(
            self,
            batch: Any,
            batch_idx: int
    ) -> Any:
        return self._get_loss(batch, prefix="val")

    def test_step(
            self,
            batch: Any,
            batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.predict(batch[0], batch[1].size(1)-1)
        error = prediction - batch[1]
        mse = error.pow(2).mean(dim=(0, 2))
        mae = error.abs().mean(dim=(0, 2))

        bounds = torch.linspace(-3.5, 3.5, 71)
        ref_dist = torch.histogramdd(
            batch[1][:, -50:].reshape(-1, 3).cpu(),
            bins=(bounds, bounds, bounds)
        ).hist
        pred_dist = torch.histogramdd(
            prediction[:, -50:].reshape(-1, 3).cpu(),
            bins=(bounds, bounds, bounds)
        ).hist
        self.test_step_scores.append({
            "mse": mse,
            "mae": mae,
            "ref_dist": ref_dist,
            "pred_dist": pred_dist,
            "n_samples": batch[0].size(0)*3
        })
        return mse[1]

    def on_test_epoch_end(self) -> None:
        # Estimate MSE and MAE
        wandb.define_metric("lead_time")
        wandb.define_metric("test/mse", step_metric="lead_time")
        wandb.define_metric("test/mae", step_metric="lead_time")
        total_samples = sum([s['n_samples'] for s in self.test_step_scores])
        mse = 0
        mae = 0
        for s in self.test_step_scores:
            weight = s['n_samples'] / total_samples
            mse = mse + s['mse'] * weight
            mae = mae + s['mae'] * weight
        all_scores = zip(mse, mae)
        for ld, scores in enumerate(all_scores):
            score_dict = {
                "test/mse": scores[0],
                "test/mae": scores[1],
                "lead_time": ld
            }
            wandb.log(score_dict)

        # Estimate Hellinger
        ref_dist = torch.stack([
            s["ref_dist"]for s in self.test_step_scores
        ], dim=0).sum(dim=0)
        pred_dist = torch.stack([
            s["pred_dist"]for s in self.test_step_scores
        ], dim=0).sum(dim=0)
        hellinger = hellinger_score(
            ref_dist/ref_dist.sum(), pred_dist/pred_dist.sum()
        )
        wandb.log({"test/hellinger": hellinger})
        wandb.run.summary["test/mse"] = mse
        wandb.run.summary["test/mae"] = mae
        wandb.run.summary["lead_time"] = torch.arange(len(mse))

    def predict(
            self,
            states: torch.Tensor,
            n_ints: int = 100
    ) -> torch.Tensor:
        trajectory = [states]
        for _ in range(n_ints):
            try:
                prediction = self(trajectory[-1])+trajectory[-1]
            except ValueError:
                prediction = torch.nan * torch.ones_like(states)
            trajectory.append(prediction)
        trajectory = torch.stack(trajectory, dim=1)
        return trajectory

    def configure_optimizers(self) -> Any:
        pass
