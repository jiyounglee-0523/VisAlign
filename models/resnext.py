import torch
from torch import nn
import torchvision
from torchvision.models import resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
from torchvision.models.resnet import Bottleneck, ResNet

class ResNextModule(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes
                 ):
        super().__init__()

        if model_name == 'resnext50_32x4d':
            self.model = resnext50_32x4d(num_classes=num_classes)

        elif model_name == 'resnext101_32x8d':
            self.model = resnext101_32x8d(num_classes=num_classes)

        elif model_name == 'resnext101_64x4d':
            self.model = resnext101_64x4d(num_classes=num_classes)

        elif model_name == 'resnext_extra':
            self.model = ResNet(block=Bottleneck, layers=[3, 4, 25, 3], num_classes=num_classes, groups=48, width_per_group=16)

    def forward(self, x):
        """
        Inputs:
            x - Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                               as a feature vector instead of a image grid.
        """

        out = self.model(x)
        return out