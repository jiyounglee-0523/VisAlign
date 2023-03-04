import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from typing import Any


class BasePostProcessor():
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, image_lists = [], [], []

        for (image_name, image) in data_loader:
            image_lists.extend(image_name.tolist())   # TODO: 확인하기
            data = image.cuda()
            # label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                # label_list.append(label[idx].cpu().tolist())

        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        # label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, image_lists