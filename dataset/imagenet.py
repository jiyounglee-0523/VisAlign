import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from PIL import Image

def transform_fn(is_training):
    if is_training:
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif not is_training:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class ImageNet(Dataset):
    def __init__(self, args, is_training=True):
        super().__init__()
        # transform
        self.transform = transform_fn(is_training)
        self.data = 'XX'

    def __len__(self):
        pass

    def __getitem__(self, item):
        image = Image.open(self.data[item]).convert('RGB')
        image = self.transform(image)
        return image


class ImageNetModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        self.train_dataset = ImageNet(self.args, is_training=True)
        self.eval_dataset = ImageNet(self.args, is_training=False)
        self.test_dataset = ImageNet(self.args, is_training=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )