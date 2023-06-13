import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from typing import Any
from tqdm import tqdm


class BasePostProcessor():
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        # conf, pred = torch.max(score, dim=1)
        return score

    def inference(self, net: nn.Module, data_loader: DataLoader):
        conf_list, image_lists = [], []

        for (image_name, image) in tqdm(data_loader):
            image_lists.extend(image_name)
            data = image.cuda()
            conf = self.postprocess(net, data)
            for idx in range(len(data)):
                conf_list.append(conf[idx].cpu().tolist())



        conf_list = np.array(conf_list)

        return conf_list, image_lists