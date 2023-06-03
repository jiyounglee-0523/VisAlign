import torch
from torch import nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import DenseNet201_Weights
from torchvision.models.densenet import DenseNet
from collections import OrderedDict

from models.utils import Identity


class DenseNetModule(nn.Module):
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

                    self.model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes)

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        elif is_ssl is True:
            
            if model_name == 'densenet201':
                self.model = densenet201(num_classes=num_classes)
            
            elif model_name == 'densenet_extra':
                growth_rate = 64
                block_config = (24, 48, 84, 64)
                num_init_features = 64

                self.model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes)

            self.hidden_dim = self.model.classifier.in_features

            # remove the last classifier
            self.model.classifier = Identity()
            # remove the last batchnorm
            # self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
            self.model.features = nn.Sequential(
                OrderedDict(
                    list(self.model.features.named_children())[:-1]
                )
            )

    def forward(self, x):
        out = self.model(x)
        return out
    
    def replace_ssl(self):
        hidden_dim = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(OrderedDict([
            ('head', nn.Linear(in_features=hidden_dim, out_features=10, bias=True))
            ]))
        features = list(self.model.features.named_children())
        features.append(('ln', nn.LayerNorm((hidden_dim,), eps=1e-06, elementwise_affine=True)))
        self.model.features = nn.Sequential(
            OrderedDict(
                self.model.features.named_children(),
                
            )
        )
        # for param in self.model.features.parameters():
        #     param.requires_grad = False