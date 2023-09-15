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


class TrajDataset(Dataset):
    def __init__(
            self,
            states_path: str,
            delta_t: int = 10,
            length: int = 100
    ):
        super().__init__()
        self.states = torch.load(states_path)
        timedelta = torch.arange(
            0, end=length*delta_t+1, step=delta_t, dtype=torch.long
        )
        init_time = torch.arange(
            0, self.states.size(1)-length*delta_t-1, dtype=torch.long
        )
        self.indexes = init_time[:, None] + timedelta[None, :]

    def __len__(self) -> int:
        return self.states.size(0) * self.indexes.size(0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_batch = idx % self.states.size(0)
        idx_index = idx // self.states.size(0)
        traj = self.states[idx_batch, self.indexes[idx_index]]
        return traj[0], traj
