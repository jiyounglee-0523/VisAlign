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

"""
'tiger.txt', 'zebra.txt', 'camel.txt', 'giraffe.txt', 'elephant.txt', 'rhino.txt',
                           'gorilla.txt', 'bear.txt', 'kangaroo.txt', 'human.txt'
"""

def return_int_label(image_name):

    label2int = {
        'tiger': 0,
        'zebra': 1,
        'camel': 2,
        'giraffe': 3,
        'elephant': 4,
        'rhino': 5,
        'gorilla': 6,
        'bear': 7,
        'kangaroo': 8,
        'human': 9,

    }

    int_label = label2int[image_name.split('-')[1]]

    return int_label



class IDImageNetTest(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

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

        file_list = ['category1.txt', 'category2.txt', 'category3.txt', 'category4.txt', 'category5.txt', 'category6.txt', 'category7.txt', 'category8.txt']

        for file in file_list:
            with open(os.path.join(dataset_path, 'label', file), 'r') as f:
                image_names = f.read().split('\n')

            image_names = image_names[:-1]
            self.data.extend(image_names)

            if file in ['category3.txt']:

                labels = [return_int_label(image_name) for image_name in image_names]
                self.target.extend(labels)

            else:
                self.target.extend([10] * len(image_names))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')[0], self.data[item].split('/')[1]

        path = os.path.join(self.dataset_path, data_source, image)

        image_data = Image.open(path).convert('RGB')
        image_data = self.transform(image_data)

        target = torch.LongTensor([self.target[item]])

        return (image, image_data, target)


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

        self.dataset_path = dataset_path
        self.data = list()

        file_list = ['category1.txt', 'category2.txt', 'category3.txt', 'category4.txt', 'category5.txt', 'category6.txt', 'category7.txt', 'category8.txt']
        # file_list = ['category3.txt']

        for file in file_list:
            with open(os.path.join(dataset_path, 'label', file), 'r') as f:
                image_names = f.read().split('\n')

            image_names = image_names[:-1]

            self.data.extend(image_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')[0], self.data[item].split('/')[1]

        path = os.path.join(self.dataset_path, data_source, image)

        image_data = Image.open(path).convert('RGB')
        image_data = self.transform(image_data)

        return (image, image_data)
