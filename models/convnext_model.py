import torch
from torch import nn
import torchvision
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision.models import ConvNeXt_Large_Weights
from torchvision.models.convnext import ConvNeXt, CNBlockConfig


class ConvNext(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes,
                 pretrained_weights,
                 freeze_weights,
                 ):
        super().__init__()

        if pretrained_weights is True:
            self.model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)

            # change the last layer to match num_classes
            self.model.classifier[-1] = nn.Linear(1536, num_classes, bias=True)

            # option to freeze weights
            if freeze_weights is True:
                for name, param in self.model.named_parameters():
                    if name.split('.')[0] != 'classifier':
                        param.requires_grad = False


        elif pretrained_weights is False:

            if model_name == 'convnext_tiny':
                self.model = convnext_tiny(num_classes=num_classes)

            elif model_name == 'convnext_small':
                self.model = convnext_small(num_classes=num_classes)

            elif model_name == 'convnext_base':
                self.model = convnext_base(num_classes=num_classes)

            elif model_name =='convnext_large':
                self.model = convnext_large(num_classes=num_classes)

            elif model_name == 'convnext_extra':
                block_setting = [
                    CNBlockConfig(192, 384, 3),
                    CNBlockConfig(384, 768, 3),
                    CNBlockConfig(768, 1536, 50),
                    CNBlockConfig(1536, None, 3),
                ]

                self.model = ConvNeXt(
                    block_setting,
                    stochastic_depth_prob=0.5,
                    num_classes=num_classes,
                )

            else:
                raise NotImplementedError


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