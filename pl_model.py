import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_model import VisionTransformerModule
from models.efficientnet_model import EfficientNetModule
from models.convnext_model import ConvNext
from mlp_mixer_pytorch import MLPMixer
from models.cnn_model import CNNNet
from models.swin_transformer_model import SwinTransformerModule
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

        elif args.model_name == 'mlp':
            self.model = MLPMixer(**self.args.mlp, **self.args.model)

        elif args.model_name == 'cnn':
            self.model = CNNNet(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
            self.model = SwinTransformerModule(model_name=args.model_name, **self.args.model)

        else:
            self.model = None
        
        assert self.model is not None

    def forward(self, x):
        return self.model(x)

    def _calculate_ce_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_ce_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_ce_loss(batch, mode="val")
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        '''
        validation_step_outputs: list
        DDP에서는 'GPU process' 별로 validation_step, validation_step_end를 거쳐 validation_step_ouptuts라는 리스트에 원소로 쌓인다.
        '''
        pass

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
        optimizer_fn = return_optimizer(self.args.optimizer)
        optimizer = optimizer_fn(self.model.parameters(), lr=self.args.lr)

        # lr scheduler
        lr_scheduler_fn = return_lr_scheduler(self.args.lr_scheduler)
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
            'monitor': 'eval_loss'
        }
