import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.imagenet import ImageNet

import pytorch_lightning as pl

from PIL import Image
import os

class ImageNetSelfSupervised(ImageNet):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, is_training=True)

        size = 224
        color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = T.Compose([T.RandomResizedCrop(size=size),
                                    T.RandomHorizontalFlip(),
                                    T.RandomApply([color_jitter], p=0.8),
                                    T.RandomGrayscale(p=0.2),
                                    T.GaussianBlur(kernel_size=int(0.1 * size)+1),
                                    T.ToTensor()])

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')

        if data_source == 'imagenet':
            path = self.dataset_path['imagenet_path']
            path = os.path.join(path, image.split('_')[0])
        elif data_source == 'celeba':
            path = self.dataset_path['celeba_path']
        elif data_source == 'giraffe':
            path = self.dataset_path['giraffe_path']
        elif data_source == 'kangaroo':
            path = self.dataset_path['kangaroo_path']
        elif data_source == 'lsp':
            path = self.dataset_path['lsp_path']
        elif data_source == 'rhino':
            path = self.dataset_path['rhino_path']
        elif data_source == 'gorilla':
            path = self.dataset_path['gorilla_path']
        else:
            print(data_source)
            raise NotImplementedError

        image = Image.open(os.path.join(path, image)).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)   # TODO: 다른거 확인하기

        target = torch.LongTensor([self.target[item]])

        return image1, image2, target


class ImageNetSSLModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        return DataLoader(
            ImageNetSelfSupervised(self.args.dataset),
            batch_size=self.args.dataset['batch_size'],
            shuffle=True,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            ImageNetSelfSupervised(self.args.dataset),
            batch_size=self.args.dataset['batch_size'],
            shuffle=False,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            ImageNetSelfSupervised(self.args.dataset),
            batch_size=self.args.dataset['batch_size'],
            shuffle=False,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True
        )