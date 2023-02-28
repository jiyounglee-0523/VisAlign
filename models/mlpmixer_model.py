import torch
from torch import nn

import timm
from mlp_mixer_pytorch import MLPMixer

class MLPMixerModule(nn.Module):
    def __init__(self,
                 args,
                 model_name,
                 num_classes,
                 pretrained_weights,
                 freeze_weights,
                 ):
        super().__init__()

        if pretrained_weights is True:
            assert model_name in ['mixer_b16_224', 'mixer_l16_224'], f'{model_name} does not have pre-trained model parameters'

            self.model = timm.create_model(model_name=model_name, pretrained=True)

            # change the last layer to mtch num_classes
            self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)

            if freeze_weights is True:
                for name, param in self.model.named_parameters():
                    if name.split('.')[0] != 'head':
                        param.requires_grad = False

        elif pretrained_weights is False:
            if model_name == 'mlp':
                self.model = MLPMixer(**args.mlp, **args.model)
            else:
                self.model = timm.create_model(model_name=model_name, pretrained=False)

                # change the last layer
                self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out