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

'''
Target label integer
'tiger.txt', 'zebra.txt', 'camel.txt', 'giraffe.txt', 'elephant.txt', 'rhino.txt',
                           'gorilla.txt', 'bear.txt', 'kangaroo.txt', 'human.txt'
'''

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
    def __init__(self, dataset_path, is_training=True):
        super().__init__()
        # transform
        self.transform = transform_fn(is_training)

        # dataset_path
        self.dataset_path = dataset_path['train'] if is_training else dataset_path['eval']

        # load sample names
        self.data = list()
        self.target = list()

        num_classes = 10

        # 1250 samples split by 9:1 ratio
        train_size = 1125
        test_size = 125

        label_file_list = ['tiger.txt', 'zebra.txt', 'camel.txt', 'giraffe.txt', 'elephant.txt', 'rhino.txt',
                           'gorilla.txt', 'bear.txt', 'kangaroo.txt', 'human.txt']

        if is_training is True:
            for i, label_file in enumerate(label_file_list):
                with open(os.path.join(dataset_path['train']['label_path'], label_file), 'r') as f:
                    image_names = f.read().split('\n')

                self.data.extend(image_names[:-1])
                self.target.extend([i] * train_size)

            assert len(self.data) == train_size * num_classes
            assert len(self.target) == train_size * num_classes

        elif is_training is False:
            for i, label_file in enumerate(label_file_list):
                with open(os.path.join(dataset_path['eval']['label_path'], label_file), 'r') as f:
                    image_names = f.read().split('\n')

                self.data.extend(image_names[:-1])
                self.target.extend([i] * test_size)

            assert len(self.data) == test_size * num_classes
            assert len(self.target) == test_size * num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')

        if data_source == 'imagenet':
            path = self.dataset_path['imagenet_path']
            path = os.path.join(path, image.split('_')[0])
        elif data_source == 'imagenet21k':
            path = self.dataset_path['imagenet21k_path']
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
        image = self.transform(image)

        target = torch.LongTensor([self.target[item]])

        return image, target


class ImageNetModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    # def prepare_data(self):
    #     self.train_dataset = ImageNet(self.args.dataset, is_training=True)
    #     self.eval_dataset = ImageNet(self.args.dataset, is_training=False)
    #     self.test_dataset = ImageNet(self.args.dataset, is_training=False)

    def train_dataloader(self):
        return DataLoader(
            ImageNet(self.args.dataset, is_training=True),
            batch_size=self.args.dataset['batch_size'],
            shuffle=True,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ImageNet(self.args.dataset, is_training=False),
            batch_size=self.args.dataset['batch_size'],
            shuffle=False,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            ImageNet(self.args.dataset, is_training=False),
            batch_size=self.args.dataset['batch_size'],
            shuffle=False,
            num_workers=self.args.dataset['dataloader_num_workers'],
            pin_memory=True,
        )