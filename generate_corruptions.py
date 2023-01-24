## Borrowed codes from ImageNet-C https://github.com/hendrycks/robustness

import torch
import numpy as np
from PIL import Image
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from tqdm import tqdm
from pkg_resources import resource_filename
import torchvision.transforms as trn

import argparse
import random

from data_generation.corruptions import *

## Image Loaders

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def return_filename(label):
    assert label in ['zebra', 'orangutan', 'hippo', 'elephant', 'tiger', 'camel', 'polar_bear', 'giraffe'], "label out of range!"
    filename = list()

    if label == 'zebra':
        filename = ['n02391049']
    elif label == 'orangutan':
        filename = ['n02480495']
    elif label == 'hippo':
        filename = ['n02398521']
    elif label == 'elephant':
        filename = ['n02504458', 'n02504013']
    elif label == 'tiger':
        filename = ['n02129604']
    elif label == 'camel':
        filename = ['n02437312']
    elif label == 'polar_bear':
        filename = ['n02134084']
    elif label == 'giraffe':
        filename = []
    return filename

def return_image_list(label):
    if label == 'giraffe':
        valid_image_list = os.listdir('/home/edlab/jylee/RELIABLE/data/animal/giraffe/data/test')
    else:
        try:
            with open(os.path.join('/home/edlab/jylee/RELIABLE/data/clean_imagenet', f'{label}.txt'), 'r') as f:
                valid_image_list = f.read().split('\n')
                valid_image_list = valid_image_list[:-1]   # remove last empty filename
        except FileNotFoundError:
            print(f'There is no file named {label}.txt in clean imagenet!')
            raise FileNotFoundError

    return valid_image_list


def return_image_name(label):
    # filename_list = return_filename(label)
    image_list = return_image_list(label)
    image_name = random.sample(image_list, 10)

    return image_name

def generate_sample(image_path, save_path, label, corruption):
    image_names = return_image_name(label)

    for severity in range(1, 11):
        image_name = image_names[severity-1]
        if label != 'giraffe':
            img = default_loader(os.path.join(image_path, image_name))
            # cropping
            bbox = bounding_box(image_name, label)
            img = img.crop(bbox)

        elif label == 'giraffe':  # skip cropping because there is no bounding box for giraffe dataset
            img = default_loader(os.path.join('/home/edlab/jylee/RELIABLE/data/animal/giraffe/data/test', image_name))
        transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.Resize(224)])

        img = transform(img)
        corrupted_img = corruption(img, severity=severity)

        # save
        save_image(corrupted_img, save_path, image_name, label, corruption.__name__, severity)


def save_image(corrupted_sample, save_path, image_name, label, corruption, severity):
    Image.fromarray(np.uint8(corrupted_sample)).save(os.path.join(save_path, f'{image_name[:-5]}-{label}-{corruption}-{severity}.jpg'), quality=85, optimize=True)



def generate_samples(image_path, save_path, number_of_samples_per_case):
    # list of functions
    corruption_lists = [gaussian_noise, shot_noise, impulse_noise, glass_blur, defocus_blur, motion_blur, zoom_blur, fog, frost, snow, contrast,
                        brightness, jpeg_compression, pixelate, elastic_transform]


    label_lists = ['zebra', 'orangutan', 'hippo', 'elephant', 'tiger', 'camel', 'giraffe', 'polar_bear']

    print(f'Total number of generated corrupted samples is {number_of_samples_per_case * (len(corruption_lists)+1) * len(label_lists) * 10}')

    print('corruptions')
    for label in tqdm(label_lists):
        for corruption in corruption_lists:
            for _ in range(number_of_samples_per_case):
                generate_sample(image_path, save_path, label, corruption)


    print('cropping')
    for label in tqdm(label_lists):
        for _ in range(number_of_samples_per_case):
            image_names = return_image_name(label)
            for i, resize in enumerate([300, 250, 200, 150, 100, 50, 40, 30, 20, 10]):
                cropped_sample, image_name = cropping(image_path, image_names[i], label, resize)
                save_image(cropped_sample, save_path, image_name, label, 'cropping', resize)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/data_storage/imagenet/ILSVRC2012_img_val')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--number_of_samples_per_case', type=int, help='number of samples to be generated in one label with one severity of one corruption')
    args = parser.parse_args()


    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    generate_samples(args.image_path, args.save_path, args.number_of_samples_per_case)


if __name__ == '__main__':
    main()