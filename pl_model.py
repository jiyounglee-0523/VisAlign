import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_model import VisionTransformerModule
from models.efficientnet_model import EfficientNetModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
import timm
from models.cnn_model import CNNNet
from models.swin_transformer_model import SwinTransformerModule
from models.resnext import ResNextModule
from utils import return_optimizer, return_lr_scheduler

import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model_name in ['vit_b_16', 'vit_l_16', 'vit_h_14', 'vit_30_16']:
            self.model = VisionTransformerModule(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_extra']:
            self.model = EfficientNetModule(model_name=args.model_name, **self.args.model)

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

        # loss = self._calculate_ce_loss(batch, mode="val")
        return output

    def validation_epoch_end(self, validation_step_outputs):
        '''
        validation_step_outputs: list
        DDP에서는 'GPU process' 별로 validation_step, validation_step_end를 거쳐 validation_step_ouptuts라는 리스트에 원소로 쌓인다.
        '''

        preds = []
        labels = []

        for val_step_output in validation_step_outputs:
            preds.append(val_step_output['preds'])
            labels.append(val_step_output['labels'])

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # print(f'val_acc: {acc}', end='\n')

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self._calculate_ce_loss(batch, mode="test")
        return loss

    def test_epoch_end(self, test_step_outputs):
        '''
        test_step_outputs: list
        DDP에서는 'GPU process' 별로 test_step, test_step_end를 거쳐 test_step_ouptuts라는 리스트에 원소로 쌓인다.
        '''
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
