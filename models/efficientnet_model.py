import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
from torchvision.models.efficientnet import MBConvConfig, EfficientNet

from functools import partial

class EfficientNetModule(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes
                 ):
        super().__init__()

        if model_name == 'efficientnet_b0':
            self.model = efficientnet_b0(num_classes=num_classes)

        elif model_name == 'efficientnet_b1':
            self.model = efficientnet_b1(num_classes=num_classes)

        elif model_name == 'efficientnet_b2':
            self.model = efficientnet_b2(num_classes=num_classes)

        elif model_name == 'efficientnet_extra':
            bneck_conf = partial(MBConvConfig, width_mult=4.1, depth_mult=4.0)

            inverted_residual_setting = [
                bneck_conf(1, 3, 1, 32, 16, 1),
                bneck_conf(6, 3, 2, 16, 24, 2),
                bneck_conf(6, 5, 2, 24, 40, 2),
                bneck_conf(6, 3, 2, 40, 80, 3),
                bneck_conf(6, 5, 1, 80, 112, 3),
                bneck_conf(6, 5, 2, 112, 192, 4),
                bneck_conf(6, 3, 1, 192, 320, 1),
            ]

            last_channel = None

            self.model = EfficientNet(inverted_residual_setting, dropout=0.5, last_channel=last_channel, num_classes=10)

        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Inputs:
            x - Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                               as a feature vector instead of a image grid.
        """

        out = self.model(x)   # [B, num_classes]
        return out