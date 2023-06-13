import torch
from torch import nn

import numpy as np

import faiss
from typing import Any
from tqdm import tqdm

from .base_postprocessor import BasePostProcessor

normalizer = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True) + 1e-10

class KNNPostProcessor(BasePostProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor['postprocessor_args']
        self.K = self.args['K']
        self.activation_log = None
        self.args_dict = self.config.postprocessor['postprocessor_sweep']

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        activation_log = []
        net.eval()

        with torch.no_grad():
            print('start setting up!')
            for images, _ in tqdm(id_loader_dict['train'], desc='Eval: ', position=0, leave=True):
                images = images.cuda()
                images = images.float()

                batch_size = images.shape[0]

                feature = net(images)

                dim = feature.shape[1]
                activation_log.append(
                    normalizer(feature.data.cpu().numpy().reshape(batch_size, dim, -1).mean(2))
                )

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature = net(data)
        feature_normed = normalizer(feature.data.cpu().numpy())

        D, _ = self.index.search(
            feature_normed,
            self.K
        )
        kth_dist = -D[:, -1]
        # _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
