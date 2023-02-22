import torch
from torch import nn

from typing import Any

from .base_postprocessor import BasePostProcessor

class MCDropoutPostProcessor(BasePostProcessor):
    def __init__(self, config):
        super().__init__(config)

        self.args = config.postprocessor.postprocessor_args
        self.dropout_times = self.args.dropout_times


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        # enable dropout
        net.eval()
        enable_dropout(net)

        logits_list = [net.forward(data) for i in range(self.dropout_times)]
        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)

        for i in range(self.dropout_times):
            logits_mean += logits_list[i]

        logits_mean /= self.dropout_times
        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)

        return pred, conf




def enable_dropout(net: nn.Module):
    """"
    Function to enable the dropout layers during test-time
    """
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()