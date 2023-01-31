import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2


class EfficientNet(nn.Module):
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