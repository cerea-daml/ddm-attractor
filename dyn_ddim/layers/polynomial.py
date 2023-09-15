#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Internal modules

main_logger = logging.getLogger(__name__)


class PolynomialLayer(torch.nn.Module):
    def __init__(self, degree: int = 2, bias: bool = True):
        super().__init__()
        self.poly_extractor = PolynomialFeatures(
            degree=degree, include_bias=bias
        )
        _ = self.poly_extractor.fit(
            X=np.ones((1, 3))
        )
        self.n_features = self.poly_extractor.n_output_features_

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_array = in_tensor.cpu().detach().numpy()
        out_array = self.poly_extractor.transform(in_array)
        out_tensor = torch.from_numpy(out_array).to(in_tensor)
        return out_tensor
