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

class IDImageNetTest(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.data = list()
        self.target = list()


        ## load dataset

    def __len__(self):
        return 0

    def __getitem__(self, item):
        image_name = self.data[item]

        image = Image.open('XX').convert('RGB')
        image = self.transform(image)

        target = torch.LongTensor([self.target[item]])

        return (image_name, image, target)


class OODImageNetTest(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.data = list()

    def __len__(self):
        return 0

    def __getitem__(self, item):
        image_name = self.data[item]

        image = Image.open('XX').convert('RGB')
        image = self.transform(image)

        return (image_name, image)
