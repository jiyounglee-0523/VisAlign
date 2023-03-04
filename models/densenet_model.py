import torch
from torch import nn
import torchvision
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import DenseNet201_Weights
from torchvision.models.densenet import DenseNet


class DenseNetModule(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes,
                 pretrained_weights,
                 freeze_weights,
                 ):
        super().__init__()

        if pretrained_weights is True:
            self.model = densenet201(weights=DenseNet201_Weights.DEFAULT)

            # change the last layer to match num_classes
            self.model.classifier = nn.Linear(self.model.classifier.in_feature, num_classes, bias=True)

            # option to freeze weights
            if freeze_weights is True:
                for name, param in self.model.named_parameters():
                    if name.split('.')[0] != "classifier":
                        param.requires_grad = False

        elif pretrained_weights is False:
            if model_name == 'densenet121':
                self.model = densenet121(num_classes=num_classes)

            elif model_name == 'densenet161':
                self.model = densenet161(num_classes=num_classes)

            elif model_name == 'densenet169':
                self.model = densenet169(num_classes=num_classes)

            elif model_name == 'densenet201':
                self.model = densenet201(num_classes=num_classes)

            elif model_name == 'densenet_extra':
                growth_rate = 64
                block_config = (24, 48, 84, 64)
                num_init_features = 64
                num_classes = 10

                self.model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes)

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.model(x)
        return out