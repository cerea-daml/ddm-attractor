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
import sys
sys.path.append('../')

import os.path
import logging
import argparse

# External modules
from tqdm import tqdm
import torch

# Internal modules
from dyn_ddim.dynamical import RK4Integrator, Lorenz63
import dyn_ddim.climatology as clim


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--n_burn_in", type=int, default=100000)
parser.add_argument("--n_ints", type=int, default=1000000)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)


def generate_l63_state(
        args: argparse.Namespace,
        n_ensemble: int = 128,
        data_type: str = "train"
) -> None:

    model = Lorenz63()
    integrator = RK4Integrator(model, dt=args.dt)
    logger.info("Initialised the model")

    initial_state = torch.randn(n_ensemble, 3) * 0.001
    for _ in tqdm(range(args.n_burn_in)):
        initial_state = integrator.integrate(initial_state)
    logger.info("Finished burn-in stage")

    trajectories = [initial_state.clone()]
    for _ in tqdm(range(args.n_ints)):
        trajectories.append(integrator.integrate(trajectories[-1]))
    trajectories = torch.stack(trajectories, dim=1)
    logger.info("Generated the trajectories")

    if data_type == 'train':
        shift = trajectories.mean(dim=(0, 1))
        scale = trajectories.std(dim=(0, 1))
        logger.info(f"mean: {shift}, std: {scale}")
    trajectories = (trajectories - torch.tensor(clim.shift)) \
                   / torch.tensor(clim.scale)
    logger.info("Normalised the trajectories")

    torch.save(
        trajectories, os.path.join(args.data_path, f"traj_{data_type:s}.pt")
    )
    logger.info("Stored the trajectories")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    generate_l63_state(args, n_ensemble=16)
    logger.info("Finished training data")
    generate_l63_state(args, n_ensemble=1, data_type="eval")
    logger.info("Finished validation data")
    generate_l63_state(args, n_ensemble=16, data_type="test")
    logger.info("Finished testing data")
