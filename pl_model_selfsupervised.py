import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_model import VisionTransformerModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
from models.swin_transformer_model import SwinTransformerModule
from models.densenet_model import DenseNetModule
from utils import return_optimizer, return_lr_scheduler

import pytorch_lightning as pl

class SSLBaseModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model_name == 'vit_30_16':
            self.model = VisionTransformerModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name == 'convnext_extra':
            self.model = ConvNext(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name == 'mlp':
            self.model = MLPMixerModule(args=args, model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['densenet201', 'densenet_extra']:
            self.model = DenseNetModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
            self.model = SwinTransformerModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        else:
            self.model = None

        assert self.model is not None

    def forward(self, x):
        pass

    def _calculate_loss(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        raise NotImplementedError