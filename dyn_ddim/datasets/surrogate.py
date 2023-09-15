#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14/07/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch
from torch.utils.data import Dataset

# Internal modules

main_logger = logging.getLogger(__name__)


class SurrogateDataset(Dataset):
    def __init__(
            self,
            states_path: str,
            delta_t: int = 10,
            corruption_level: float = 0.0
    ):
        super().__init__()
        self.corruption_level = corruption_level
        self.delta_t = delta_t
        self.states = torch.load(states_path).reshape(-1, 3)

    def __len__(self) -> int:
        return self.states.shape[0]-self.delta_t

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        initial = self.states[idx]
        diff = self.states[idx+self.delta_t] - initial
        diff = diff + self.corruption_level * torch.randn_like(diff)
        return initial, diff
