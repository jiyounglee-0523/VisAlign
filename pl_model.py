import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_model import VisionTransformer
from models.efficientnet_model import EfficientNet
from models.convnext_model import ConvNext

import pytorch_lightning as pl

# model import
from utils import return_optimizer, return_lr_scheduler

class BaseModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model_name == 'vit':
            self.model = VisionTransformer(**self.args.vit, **self.args.model)

        elif args.model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            self.model = EfficientNet(model_name=args.model_name, **self.args.model)

        elif args.model_name in ['convnext_tiny', 'convnext_small', 'convnext_base']:
            self.model = ConvNext(model_name=args.model_name, **self.args.model)

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
