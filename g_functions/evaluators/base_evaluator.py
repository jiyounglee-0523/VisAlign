import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import os
from tqdm import tqdm

from g_functions.postprocessors.base_postprocessor import BasePostProcessor


def to_np(x):
    return x.data.cpu().numpy()


class BaseEvaluator():
    def __init__(self, config):
        self.config = config

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostProcessor = None,
                 epoch_idx: int = -1
                 ):
        net.eval()

        loss_avg = 0.0
        correct = 0

