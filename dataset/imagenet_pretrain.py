import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from PIL import Image
import os

class ImageNetPretrainModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        return DataLoader(
            torchvision.datasets.ImageFolder(
                '/home/data_storage/imagenet/train',
                T.Compose([
                    T.Resize(256),
                    T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            ),
            batch_size=self.args.dataset['batch_size'],
            shuffle=True,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            torchvision.datasets.ImageFolder(
                '/home/data_storage/imagenet/ILSVRC2012_img_val_org',
                T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            ),
            batch_size=self.args.dataset['batch_size'],
            shuffle=False,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True
        )

    def test_dataloader(self):
        pass
