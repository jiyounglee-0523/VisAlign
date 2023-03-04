import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import os

from models.vit_model import VisionTransformerModule
from models.efficientnet_model import EfficientNetModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
from models.cnn_model import CNNNet
from models.swin_transformer_model import SwinTransformerModule
from models.resnext import ResNextModule
from dataset.imagenet_test import IDImageNetTest, OODImageNetTest
from dataset.imagenet import ImageNet
from g_functions.utils import get_postprocessor

def return_model(model_name):
    num_classes = 10,
    pretrained_weights = False,
    freeze_weights = False

    if model_name in ['vit_b_16', 'vit_l_16', 'vit_h_14', 'vit_30_16']:
        model = VisionTransformerModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_extra']:
        model = EfficientNetModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_extra']:
        model = ConvNext(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['mixer_s16_224', 'mixer_s32_224', 'mixer_b16_224', 'mixer_b32_224', 'mixer_l16_224',
                             'mixer_l32_224', 'mlp']:
        if model_name == 'mlp':
            raise NotImplementedError
        model = MLPMixerModule(args=None, model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name == 'cnn':
        model = CNNNet(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
        model = SwinTransformerModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext_extra']:
        model = ResNextModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    else:
        raise NotImplementedError

    return model


def return_weight_loaded_model(model_name, ckpt_path):
    model = return_model(model_name)
    # load ckpt file
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def classify_main(args):
    id_dataloader = DataLoader(
        IDImageNetTest(args.dataset_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = return_weight_loaded_model(args.model_name, args.ckpt_path)
    model.eval()

    image_names = []
    gt_targets = []
    predictions = []

    for (image_name, image, target) in id_dataloader:
        image = image.cuda()

        # forward
        output = model(image)
        prediction = output.argmax(dim=-1)  # prediction

        image_names.append(image_name)
        gt_targets.append(target)
        predictions.append(prediction)

    ## 후처리해주기

    # dictionary 만들고 저장하기


def OOD_main(args):
    ood_dataloader = DataLoader(
        OODImageNetTest(args.dataset_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    dataset_path = {
        'train': {
            'label_path': '/home/edlab/jylee/RELIABLE/data/final_dataset/final_train',
            'imagenet_path': '/home/data_storage/imagenet/train',
            'celeba_path': '/home/data_storage/CelebA/celeba/img_align_celeba',
            'giraffe_path': '/home/edlab/jylee/RELIABLE/data/animal/giraffe/data',
            'kangaroo_path': '/home/edlab/jylee/RELIABLE/data/animal/kangaroo/data',
            'lsp_path': '/home/edlab/jylee/RELIABLE/data/LSP/train',
            'rhino_path': '/home/edlab/jylee/RELIABLE/data/animal/rhino/data',
            'gorilla_path': '/home/edlab/jylee/RELIABLE/data/animal/gorilla/data',
        },
        'eval': {
            'label_path': '/home/edlab/jylee/RELIABLE/data/final_dataset/final_eval',
            'imagenet_path': '/home/data_storage/imagenet/train',
            'celeba_path': '/home/data_storage/CelebA/celeba/img_align_celeba',
            'giraffe_path': '/home/edlab/jylee/RELIABLE/data/animal/giraffe/data',
            'kangaroo_path': '/home/edlab/jylee/RELIABLE/data/animal/kangaroo/data',
            'lsp_path': '/home/edlab/jylee/RELIABLE/data/LSP/train',
            'rhino_path': '/home/edlab/jylee/RELIABLE/data/animal/rhino/data',
            'gorilla_path': '/home/edlab/jylee/RELIABLE/data/animal/gorilla/data',
        }
    }

    id_loader_dict = {
        'train': DataLoader(
            ImageNet(dataset_path, is_training=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True),

        'val': DataLoader(ImageNet(dataset_path, is_training=False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True)
    }

    # model
    model = return_weight_loaded_model(args.model_name, args.ckpt_path)
    model.eval()

    postprocessor = get_postprocessor(args.postprocessor_name)
    postprocessor.setup(model, id_loader_dict, ood_dataloader)

    # evaluate
    metrics_list = list()

    ood_pred, ood_conf, image_lists = postprocessor.inference(model, ood_dataloader)

    # 후처리
    # 저장하기

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--model_name')
    parser.add_argument('--batch_size', default=16, type=int)

    args = parser.parse_args()

    assert args.ckpt_dir is not None, 'Please specify checkpoint file path'

    # ID Dataset classification
    classify_main(args)

    # OOD score
    OOD_main(args)


if __name__ == '__main__':
    main()