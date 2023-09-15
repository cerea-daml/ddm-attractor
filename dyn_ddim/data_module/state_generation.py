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
from typing import Optional
import os.path

# External modules
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Internal modules


logger = logging.getLogger(__name__)


class GenerationDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = torch.load(
            os.path.join(self.data_path, f"traj_train.pt")
        ).view(-1, 3)
        self.eval_dataset = torch.load(
            os.path.join(self.data_path, f"traj_eval.pt")
        ).view(-1, 3)
        self.test_dataset = torch.load(
            os.path.join(self.data_path, f"traj_test.pt")
        ).view(-1, 3)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            multiprocessing_context='fork'
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            multiprocessing_context='fork'
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            multiprocessing_context='fork'
        )
