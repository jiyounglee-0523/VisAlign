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
    assert label in ['fish', 'bear', 'boat', 'cat', 'bottle', 'truck', 'bird', 'dog'], "label out of range!"
    filename = list()

    if label == 'fish':
        filename = ['n01440764', 'n01443537', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072',
         'n02640242', 'n02641379', 'n02643566', 'n02655020']
    elif label == 'bear':
        filename = ['n02132136', 'n02133161', 'n02134084', 'n02134418']
    elif label == 'boat':
        filename = ['n02951358', 'n03344393', 'n03662601', 'n04273569', 'n04612373', 'n04612504']
    elif label == 'cat':
        filename = ["n02122878", "n02123045", "n02123159", "n02126465", "n02123394", "n02123597",
            "n02124075", "n02125311"]
    elif label == 'bottle':
        filename = ['n02823428', 'n03937543', 'n03983396','n04557648', 'n04560804', 'n04579145', 'n04591713']
    elif label == 'truck':
        filename = ['n03345487', 'n03417042', 'n03770679','n03796401', 'n00319176', 'n01016201', 'n03930630', 'n03930777', 'n05061003',
            'n06547832', 'n10432053', 'n03977966', 'n04461696', 'n04467665']
    elif label == 'bird':
        filename = ['n01321123', 'n01514859', 'n01792640', 'n07646067', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01562265',
            'n01560419', 'n01582220', 'n10281276', 'n01592084', 'n01601694', 'n01614925', 'n01616318', 'n01622779', 'n01795545', 'n01796340', 'n01797886', 'n01798484',
            'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01855032', 'n01855672',
            'n07646821', 'n01860187', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02013706', 'n02017213', 'n02018207',
            'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570']
    elif label == 'dog':
        filename = ['n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364',
            'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467',
            'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258',
            'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298',
            'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735',
            'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056',
            'n02105162', 'n02105251', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312',
            'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185',
            'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n08825211', 'n02111500', 'n02112018', 'n02112350', 'n02112706', 'n02113023', 'n02113624',
            'n02113712', 'n02113799', 'n02113978']
    return filename

def return_image_list(filename_list):
    with open('/home/data_storage/imagenet/valid.txt', 'r') as f:
        valid_image_list = f.readlines()

    valid_image_list = [v[:-3] for v in valid_image_list]
    valid_image_list = [v.split(' ')[0] for v in valid_image_list if v.split(' ')[1] in filename_list]
    return valid_image_list

def return_image_name(image_path:str, label):
    filename_list = return_filename(label)
    image_list = return_image_list(filename_list)
    image_name = random.choice(image_list)

    # image_list = os.listdir(os.path.join(image_path, image_name))
    # image_name = random.choice(image_list)
    return image_name

def generate_sample(image_path, label, corruption, severity):
    image_name = return_image_name(image_path, label)

    img = default_loader(os.path.join(image_path, image_name))
    transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.Resize(224)])
    img = transform(img)
    corrupted_img = corruption(img, severity=severity)
    return corrupted_img, image_name

def save_image(corrupted_sample, save_path, image_name, label, corruption, severity):
    Image.fromarray(np.uint8(corrupted_sample)).save(os.path.join(save_path, f'{image_name[:-5]}_{label}_{corruption}_{severity}.jpg'), quality=85, optimize=True)



def generate_samples(image_path, save_path, number_of_samples_per_case):
    # list of functions
    corruption_lists = [gaussian_noise, shot_noise, impulse_noise, glass_blur, defocus_blur, motion_blur, zoom_blur, fog, frost, snow, contrast,
                        brightness, jpeg_compression, pixelate, elastic_transform]


    label_lists = ['dog', 'fish', 'bear', 'boat', 'cat', 'bottle', 'truck', 'bird']

    print(f'Total number of generated corrupted samples is {number_of_samples_per_case * (len(corruption_lists)+1) * len(label_lists) * 10}')

    for label in label_lists:
        for corruption in corruption_lists:
            for severity in range(1, 11):
                for _ in range(number_of_samples_per_case):
                    corrupted_sample, image_name = generate_sample(image_path, label, corruption, severity=severity)
                    save_image(corrupted_sample, save_path, image_name, label, corruption.__name__, severity)

    for label in label_lists:
        for resize in [300, 250, 200, 150, 100, 50, 40, 30, 20, 10]:
            for _ in range(number_of_samples_per_case):
                image_name = return_image_name(image_path, label)
                cropped_sample, image_name = cropping(image_path, image_name, label, resize)
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