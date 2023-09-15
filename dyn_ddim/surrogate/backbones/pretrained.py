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
from typing import Iterable

# External modules
import torch.nn

# Internal modules
from dyn_ddim.denoising_diffusion import DenoisingDiffusionModule
from dyn_ddim.layers import FilmLayer

main_logger = logging.getLogger(__name__)


class StaticFilm(torch.nn.Module):
    def __init__(self, old_layer: FilmLayer, time_embedding: torch.Tensor):
        super().__init__()
        with torch.no_grad():
            scale, shift = old_layer.affine_film(
                time_embedding
            ).chunk(2, dim=-1)
        self.register_parameter("scale", torch.nn.Parameter(scale[None, ...]))
        self.register_parameter("shift", torch.nn.Parameter(shift[None, ...]))

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        return in_tensor * (self.scale + 1) + self.shift


def replace_film(model, time_embedding):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_film(module, time_embedding)

        if isinstance(module, FilmLayer):
            static_film = StaticFilm(
                module, time_embedding
            )
            setattr(model, n, static_film)


class PretrainedBackbone(torch.nn.Module):
    def __init__(
            self,
            ckpt_path: str = None,
            times_to_use: Iterable[int] = (500, 250, 100, 50, 0),
            ema: bool = False,
            untrained: bool = False
    ):
        super().__init__()
        if untrained:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            hyper_params = checkpoint[
                DenoisingDiffusionModule.CHECKPOINT_HYPER_PARAMS_KEY
            ]
            parent_module = DenoisingDiffusionModule(**hyper_params)
        else:
            parent_module = DenoisingDiffusionModule.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )

        self.example_input_array = torch.randn(1, 3)
        self.network = parent_module.denoising_network
        if ema and (not untrained):
            self.load_ema_weights(self.network, ckpt_path)
        self.make_film_static(self.network, times_to_use)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @staticmethod
    def load_ema_weights(
            network: torch.nn.Module,
            ckpt_path: str
    ) -> None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        ema_params = state_dict["callbacks"]["EMA"]["params"]
        param_dict = dict(
            zip(network.state_dict().keys(), ema_params)
        )
        network.load_state_dict(param_dict)

    @staticmethod
    def make_film_static(model, times_to_use):
        time_tensor = torch.tensor(times_to_use)[:, None].float()
        time_embedding = model.embedding_layer(time_tensor)
        replace_film(model, time_embedding)
        model.embedding_layer = torch.nn.Identity()

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        dummy_time = torch.ones(
            1, 1, device=in_tensor.device, layout=in_tensor.layout,
            dtype=in_tensor.dtype
        )
        features = self.network.extract_features(
            in_tensor[..., None, :], dummy_time
        )
        return features.view(*in_tensor.shape[:-1], -1)
