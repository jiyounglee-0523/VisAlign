import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_model import VisionTransformerModule
from models.efficientnet_model import EfficientNetModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
from models.cnn_model import CNNNet
from models.swin_transformer_model import SwinTransformerModule
from models.resnext import ResNextModule
from utils import return_optimizer, return_lr_scheduler

import pytorch_lightning as pl

class SSLBaseModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model_name in ['vit_b_16', 'vit_l_16', 'vit_h_14', 'vit_30_16']:
            self.model = VisionTransformerModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_extra']:
            self.model = EfficientNetModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
                                 'convnext_extra']:
            self.model = ConvNext(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['mixer_s16_224', 'mixer_s32_224', 'mixer_b16_224', 'mixer_b32_224', 'mixer_l16_224',
                                 'mixer_l32_224', 'mlp']:
            self.model = MLPMixerModule(args=args, model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name == 'cnn':
            self.model = CNNNet(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
            self.model = SwinTransformerModule(model_name=args.model_name, is_ssl=True, **self.args.model)

        elif args.model_name in ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext_extra']:
            self.model = ResNextModule(model_name=args.model_name, is_ssl=True, **self.args.model)

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
        # optimizer
        optimizer_fn = return_optimizer(self.args.trainer['optimizer'])
        optimizer = optimizer_fn(self.model.parameters(), lr=float(self.args.trainer['lr']))

        # lr scheduler
        lr_scheduler_fn = return_lr_scheduler(self.args.trainer['lr_scheduler'])
        scheduler = lr_scheduler_fn(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }