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
from models.densenet_model import DenseNetModule
from utils import return_optimizer, return_lr_scheduler

import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.validation_step_outputs = list()

        if args.model_name in ['vit_b_16', 'vit_l_16', 'vit_h_14', 'vit_30_16']:
            self.model = VisionTransformerModule(model_name=args.model_name, is_ssl=args.cont_ssl, **self.args.model)

        elif args.model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_extra']:
            self.model = EfficientNetModule(model_name=args.model_name, is_ssl=args.cont_ssl, **self.args.model)

        elif args.model_name in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_extra']:
            self.model = ConvNext(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['mixer_s16_224', 'mixer_s32_224', 'mixer_b16_224', 'mixer_b32_224', 'mixer_l16_224', 'mixer_l32_224', 'mlp']:
            self.model = MLPMixerModule(args=args, model_name=args.model_name, **self.args.model)

        elif args.model_name == 'cnn':
            self.model = CNNNet(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
            self.model = SwinTransformerModule(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext_extra']:
            self.model = ResNextModule(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet_extra']:
            self.model = DenseNetModule(model_name=args.model_name, **self.args.model)

        else:
            self.model = None
        
        assert self.model is not None

    def forward(self, x):
        return self.model(x)

    def _calculate_ce_loss(self, batch, mode="train"):
        imgs, labels = batch
        labels = labels.squeeze(-1)
        preds = self.model(imgs)

        loss = F.cross_entropy(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss, sync_dist=True)
        self.log("%s_acc" % mode, acc, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.squeeze(-1)
        preds = self.model(imgs)

        loss = F.cross_entropy(preds, labels)

        # log
        self.log('train_loss', loss, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.squeeze(-1)
        preds = self.model(imgs)

        output = {
            'preds': preds,
            'labels': labels,
        }

        self.validation_step_outputs.append(output)

        # loss = self._calculate_ce_loss(batch, mode="val")
        return output

    def on_validation_epoch_end(self):

        preds = []
        labels = []

        for val_step_output in self.validation_step_outputs:
            preds.append(val_step_output['preds'])
            labels.append(val_step_output['labels'])

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self._calculate_ce_loss(batch, mode="test")
        return loss

    def on_test_epoch_end(self):
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
            'monitor': 'val_loss'
        }
