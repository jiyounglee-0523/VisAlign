import torch
from torch import nn
import torchvision
from torchvision.models import resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
from torchvision.models import ResNeXt101_32X8D_Weights
from torchvision.models.resnet import Bottleneck, ResNet

class ResNextModule(nn.Module):
    def __init__(self,
                 model_name,
                 is_ssl=False,
                 num_classes=10,
                 pretrained_weights=False,
                 freeze_weights=False,
                 ):
        super().__init__()

        if is_ssl is False:
            if pretrained_weights is True:
                self.model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)

                # change the last layer to match num_classes
                self.model.fc = nn.Linear(2048, num_classes, bias=True)

                # option to freeze weights
                if freeze_weights is True:
                    for name, param in self.model.named_parameters():
                        if name.split('.')[0] != 'fc':
                            param.requires_grad = False


            elif pretrained_weights is False:
                if model_name == 'resnext50_32x4d':
                    self.model = resnext50_32x4d(num_classes=num_classes)

                elif model_name == 'resnext101_32x8d':
                    self.model = resnext101_32x8d(num_classes=num_classes)

                elif model_name == 'resnext101_64x4d':
                    self.model = resnext101_64x4d(num_classes=num_classes)

                elif model_name == 'resnext_extra':
                    self.model = ResNet(block=Bottleneck, layers=[3, 4, 25, 3], num_classes=num_classes, groups=48, width_per_group=16)

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        elif is_ssl is True:
            self.model = ResNet(block=Bottleneck, layers=[3, 4, 25, 3], num_classes=num_classes, groups=48, width_per_group=16)

            # remove the last classifier and maxpool
            self.model = nn.Sequential(*list(self.model.children())[:-2])


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