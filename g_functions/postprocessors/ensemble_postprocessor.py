import torch
from torch import nn

import numpy as np

from typing import Any
from copy import deepcopy

from .base_postprocessor import BasePostProcessor

class EnsemblePostprocessor(BasePostProcessor):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.postprocess_config = config.postprocessor
        self.postprocess_args = self.postprocess_config['postprocessor_args']
        self.num_networks = self.postprocess_args['num_networks']


    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    def postprocess(self, net: nn.Module, data: Any):
        logits_list = [
            net[i](data) for i in range(self.num_networks)
        ]
        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.num_networks):
            logits_mean += logits_list[i]
        logits_mean /= self.num_networks

        score = torch.softmax(logits_mean, dim=1)
        conf, _ = torch.max(score, dim=1)

        return conf