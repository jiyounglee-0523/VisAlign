import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import os
import pickle
from tqdm import tqdm
import yaml

from models.vit_model import VisionTransformerModule
from models.efficientnet_model import EfficientNetModule
from models.convnext_model import ConvNext
from models.densenet_model import DenseNetModule
from models.mlpmixer_model import MLPMixerModule
from models.cnn_model import CNNNet
from models.swin_transformer_model import SwinTransformerModule
from models.resnext import ResNextModule
from dataset.imagenet_test import IDImageNetTest, OODImageNetTest
from dataset.imagenet import ImageNet
from g_functions.utils import get_postprocessor
from models.utils import Identity

def return_model(model_name, args=None):
    num_classes = 10
    pretrained_weights = False
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
            model = MLPMixerModule(args=args, model_name='mixer_b16_224', num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)
        else:
            model = MLPMixerModule(args=None, model_name='mixer_l16_224', num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name == 'cnn':
        model = CNNNet(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['swin_t', 'swin_s', 'swin_b', 'swin_extra']:
        model = SwinTransformerModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext_extra']:
        model = ResNextModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    elif model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet_extra']:
        model = DenseNetModule(model_name=model_name, num_classes=num_classes, pretrained_weights=pretrained_weights, freeze_weights=freeze_weights)

    else:
        raise NotImplementedError

    return model


def return_weight_loaded_model(model_name, ckpt_path, args=None):
    model = return_model(model_name, args)
    # load ckpt file
    ckpt = torch.load(ckpt_path)
    ckpt['state_dict'] = {k[6:]: v for k, v in ckpt['state_dict'].items()}
    # if model_name == 'densenet201':
    #     ckpt['state_dict']['model.classifier.weight'] = ckpt['state_dict'].pop('model.classifier.head.weight')
    #     ckpt['state_dict']['model.classifier.bias'] = ckpt['state_dict'].pop('model.classifier.head.bias')
    model.load_state_dict(ckpt['state_dict'])
    return model


def model_feature_list(model_name, model):
    if model_name in ['vit_b_16', 'vit_l_16', 'vit_h_14', 'vit_30_16']:
        model.model.heads = Identity()
        model.model.encoder = nn.Sequential(*list(model.model.encoder.children())[:-1])

    elif model_name in ['convnext_extra']:
        model.classifier = Identity()

    elif model_name in ['swin_extra']:
        model.model.head = Identity()

    # elif model_name in ['mlp']:
    #     model = nn.Sequential(*list(model.model.children())[:-1])

    return model


def classify_main(args):
    id_dataloader = DataLoader(
        IDImageNetTest(args.test_dataset_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = return_weight_loaded_model(args.model_name, args.ckpt_dir, args=args)
    model.eval()
    model = model.cuda()

    image_names = []
    gt_targets = []
    predictions = []

    with torch.no_grad():
        for (image_name, image, target) in tqdm(id_dataloader):
            image = image.cuda()

            # forward
            prediction = model(image)
            prediction = F.softmax(prediction, dim=-1)

            # target
            target = F.one_hot(target, num_classes=11)

            image_names.extend(image_name)
            gt_targets.append(target)
            predictions.append(prediction)

    gt_targets = torch.vstack(gt_targets)
    predictions = torch.vstack(predictions)

    id_prediction = dict()

    for image_name, gt_target, prediction in zip(image_names, gt_targets, predictions):
        id_prediction[image_name] = (gt_target, prediction)

    with open(os.path.join(args.save_dir, f'{args.model_name}-{args.seed}-id_prediction.pk'), 'wb') as f:
        pickle.dump(id_prediction, f)

    print('Finished Predictiong In-Domain Dataset!')

def OOD_main(args):
    ood_dataloader = DataLoader(
        OODImageNetTest(
            args.test_dataset_path
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    dataset_path = {
        'train': {
            'label_path': f'{args.train_dataset_path}/train_split_filename/final_train',
            'imagenet21k_path': f'{args.train_dataset_path}/train_files',
        },
        'eval': {
            'label_path': f'{args.train_dataset_path}/train_split_filename/final_eval',
            'imagenet21k_path': f'{args.train_dataset_path}/train_files',
        }
    }

    id_loader_dict = {
        'train': DataLoader(
            ImageNet(dataset_path, is_training=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True),

        'val': DataLoader(
            ImageNet(dataset_path, is_training=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)
    }

    # model
    model = return_weight_loaded_model(args.model_name, args.ckpt_dir, args=args)

    if args.postprocessor_name in ['knn', 'mds', 'tapudd']:
        model = model_feature_list(args.model_name, model)

    model.eval()
    model = model.cuda()

    postprocessor = get_postprocessor(args.postprocessor_name)(args)
    postprocessor.setup(model, id_loader_dict, ood_dataloader)

    print('Start Calculating OOD Score')
    ood_conf, image_lists = postprocessor.inference(model, ood_dataloader)

    ood_score = dict()
    for ood, image in zip(ood_conf, image_lists):
        ood_score[image] = ood

    with open(os.path.join(args.save_dir, f'{args.model_name}-{args.postprocessor_name}-{args.seed}-oodscore.pk'), 'wb') as f:
        pickle.dump(ood_score, f)

    print('Finished Calculating OOD Score')

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default='./', type=str, help='')
    parser.add_argument('--config', default='./config/ood_postprocessors', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--model_name')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_dataset_path', default='./VisAlign_dataset/open_test_set', type=str)
    parser.add_argument('--train_dataset_path', default='./VisAlign_dataset')
    parser.add_argument('--postprocessor_name', type=str, choices=['knn', 'mcdropout', 'mds', 'odin', 'msp', 'tapudd'])
    parser.add_argument('--seed', type=int, default=45)

    args = parser.parse_args()

    assert args.ckpt_dir is not None, 'Please specify checkpoint file path'
    # ID Dataset classification
    classify_main(args)

    OOD_main(args)


if __name__ == '__main__':
    main()