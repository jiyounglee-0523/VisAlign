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

def return_int_label(file, image_name, valid_dict):
    wordnet_label_dict = {
        'n02129604': 'tiger',
        'n02391049': 'zebra',
        'n02504458': 'elephant',
        'n02504013': 'elephant',
        'n02437312': 'camel',
        'n01877134': 'kangaroo',
        'n02510455': 'bear',
        'n02134084': 'bear',
        'n02134418': 'bear',
        'n02132136': 'bear',
        'n02133161': 'bear',
        'n02480855': 'gorilla',
    }

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

    int_label = None

    if file == 'in-domain.txt':
        if image_name.split('/')[0] != 'imagenet':
            int_label = label2int[image_name.split('/')[0]]
        else:
            wordnet_label = valid_dict[image_name.split('/')[1]]
            int_label = label2int[wordnet_label_dict[wordnet_label]]

    elif file == 'uncommon_features.txt':
        int_label = label2int[wordnet_label_dict[image_name.split('_')[1]]]

    elif file == 'category3.txt':
        int_label = label2int[image_name.split('-')[1]]

    elif file == 'renditions.txt':
        int_label = label2int[image_name.split('/')[6]]

    elif file in ['low_corruptions.txt', 'high_corruptions.txt']:
        int_label = label2int[image_name.split('-')[1]]

    else:
        raise NotImplementedError

    return int_label



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

        file_list = ['category1.txt', 'category2.txt', 'category3.txt', 'category4.txt', 'category5.txt', 'category6.txt', 'category7.txt', 'category8.txt']
        # file_list = ['category3.txt']

        with open(os.path.join('/home/data_storage/imagenet/valid.txt'), 'r') as f:
            valid_list = f.read().split('\t\n')

        valid_dict = {v.split(' ')[0]: v.split(' ')[1][:-1] for v in valid_list[:-1]}

        for file in file_list:
            with open(os.path.join(dataset_path, file), 'r') as f:
                image_names = f.read().split('\n')

            image_names = image_names[:-1]
            self.data.extend(image_names)

            if file in ['category3.txt']:

                labels = [return_int_label(file, image_name, valid_dict) for image_name in image_names]
                self.target.extend(labels)

            else:
                self.target.extend([10] * len(image_names))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')[0], self.data[item].split('/')[1]

        if data_source == 'imagenet':
            path = os.path.join('/home/data_storage/imagenet/ILSVRC2012_img_val', image)
        elif data_source == 'adversarial':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/adversarial', image)
        elif data_source == 'bgchallenge':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/04_carnivore', image)
        elif data_source == 'giraffe':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'kangaroo':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'rhino':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'gorilla':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'camel':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'domainnet':
            path = os.path.join('/'+self.data[item].split('//')[1])
            image = self.data[item].split('/')[-1]
        elif data_source == 'bgchallengprimate':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/07_primate', image)
        elif data_source == 'bgchallengcarnivore':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/04_carnivore', image)
        elif data_source == 'multiple_stablediffusion':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/multiple_characteristics/StableDiffusion', image)
        elif data_source == 'corruptions':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/corrupted_images', image)
        elif data_source == 'diffusionuncommon':
            path = os.path.join('/home/edlab/shkim/RELIABLE/data/unfamiliar_bg', image)
        elif data_source == 'ivs':
            path = os.path.join('/home/jylee/RELIABILITY/competition_dataset/validation/category7_v2', image)
        elif data_source == 'fts':
            path = os.path.join('/home/jylee/RELIABILITY/competition_dataset/test/category7_v2', image)
        elif data_source == 'refinedcategory':
            path = os.path.join('/home/edlab/shkim/RELIABLE/data/cat2_sd_100_100', image)
        elif 'category' in data_source:
            path = os.path.join(f'/home/jylee/RELIABILITY/competition_dataset/test/{data_source}', image)
        else:
            print(self.data[item])
            raise NotImplementedError


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

        self.data = list()

        file_list = ['category1.txt', 'category2.txt', 'category3.txt', 'category4.txt', 'category5.txt', 'category6.txt', 'category7.txt', 'category8.txt']
        # file_list = ['category3.txt']

        for file in file_list:
            with open(os.path.join(dataset_path, file), 'r') as f:
                image_names = f.read().split('\n')

            image_names = image_names[:-1]

            self.data.extend(image_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_source, image = self.data[item].split('/')[0], self.data[item].split('/')[1]

        if data_source == 'imagenet':
            path = os.path.join('/home/data_storage/imagenet/ILSVRC2012_img_val', image)
        elif data_source == 'adversarial':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/adversarial', image)
        elif data_source == 'bgchallenge':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/04_carnivore', image)
        elif data_source == 'giraffe':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'kangaroo':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'rhino':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'gorilla':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'camel':
            path = os.path.join(f'/home/edlab/jylee/RELIABLE/data/animal/{data_source}/data/test', image)
        elif data_source == 'domainnet':
            path = os.path.join('/' + self.data[item].split('//')[1])
            image = self.data[item].split('/')[-1]
        elif data_source == 'bgchallengprimate':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/07_primate', image)
        elif data_source == 'bgchallengcarnivore':
            path = os.path.join('/home/data_storage/BackgroundChallenge/bg_challenge/mixed_rand/val/04_carnivore', image)
        elif data_source == 'multiple_stablediffusion':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/multiple_characteristics/StableDiffusion', image)
        elif data_source == 'corruptions':
            path = os.path.join('/home/edlab/jylee/RELIABLE/data/corrupted_images', image)
        elif data_source == 'diffusionuncommon':
            path = os.path.join('/home/edlab/shkim/RELIABLE/data/unfamiliar_bg', image)
        elif data_source == 'ivs':
            path = os.path.join('/home/jylee/RELIABILITY/competition_dataset/validation/category7_v2', image)
        elif data_source == 'fts':
            path = os.path.join('/home/jylee/RELIABILITY/competition_dataset/test/category7_v2', image)
        elif data_source == 'refinedcategory':
            path = os.path.join('/home/edlab/shkim/RELIABLE/data/cat2_sd_100_100', image)
        elif 'category' in data_source:
            path = os.path.join(f'/home/jylee/RELIABILITY/competition_dataset/test/{data_source}', image)
        else:
            raise NotImplementedError

        image_data = Image.open(path).convert('RGB')
        image_data = self.transform(image_data)

        return (image, image_data)
