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
from torch.utils.data import random_split

# Internal modules
from ..datasets import SurrogateDataset, TrajDataset


logger = logging.getLogger(__name__)


class SurrogateDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            n_train_samples: int = None,
            corruption_level: float = 0.0,
            train_batch_size: int = 16384,
            val_batch_size: int = 16384,
            delta_t: int = 10,
            n_test_ints: int = 100,
            num_workers: int = 0,
            pin_memory: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.delta_t = delta_t
        self.corruption_level = corruption_level
        self.n_train_samples = n_train_samples
        if n_train_samples is not None:
            self.train_batch_size = min(train_batch_size, n_train_samples)

        self.n_test_ints = n_test_ints
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = SurrogateDataset(
            os.path.join(self.data_path, f"traj_train.pt"),
            delta_t=self.delta_t, corruption_level=self.corruption_level
        )
        if self.n_train_samples is None:
            train_samples = len(train_dataset)
        else:
            train_samples = self.n_train_samples
        self.train_dataset, _ = random_split(
            train_dataset,
            [train_samples, len(train_dataset)-train_samples]
        )
        self.eval_dataset = SurrogateDataset(
            os.path.join(self.data_path, f"traj_eval.pt"),
            delta_t=self.delta_t, corruption_level=0.0
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.train_batch_size,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            multiprocessing_context='fork'
        )
