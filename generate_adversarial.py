## Generate Adversarial Samples by FGSM (borrowed code from https://tutorials.pytorch.kr/beginner/fgsm_tutorial.html)

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torchvision.models import \
    resnet50, ResNet50_Weights, \
    convnext_base, ConvNeXt_Base_Weights, \
    densenet121, DenseNet121_Weights, \
    efficientnet_b0, EfficientNet_B0_Weights, \
    resnext50_32x4d, ResNeXt50_32X4D_Weights, \
    vgg16, VGG16_Weights, \
    wide_resnet50_2, Wide_ResNet50_2_Weights

import numpy as np

import os
import argparse
import random
import json
from PIL import Image

# unused
def return_wordnet_label(label):
    assert label in ['tiger', 'zebra', 'camel', 'elephant', 'bear', 'kangaroo', 'giraffe', 'rhino', 'human', 'gorilla'],  'label outside of the candidates'

    if label == 'tiger':
        wordnet_label = ['n02129604']
    elif label == 'zebra':
        wordnet_label = ['n02391049']
    elif label == 'camel':
        wordnet_label = ['n02437312']
    elif label == 'elephant':
        wordnet_label = ['n02504458', 'n02504013']
    elif label == 'hippo':
        wordnet_label = ['n02398521']
    elif label == 'orangutan':
        wordnet_label = ['n02480495']
    elif label == 'polar_bear':
        wordnet_label = ['n02134084']
    elif label == 'kangaroo':
        wordnet_label = ['n01877812']
    else:
        raise NotImplementedError

    return wordnet_label


def return_metadata():
    with open('/home/data_storage/imagenet/imagenet_class_index.json', 'r') as f:
        metadata = json.load(f)

    metadata = {v[0]: k for k, v in metadata.items()}

    return metadata


def return_image_list():
    # load ImageNet validation samples
    with open('/home/data_storage/imagenet/valid.txt', 'r') as f:
        valid_samples = f.read().split('\n')
        valid_samples = valid_samples[:-1]

    valid_samples = {v.split(' ')[0]: v.split(' ')[1][:9] for v in valid_samples}

    return valid_samples


def filter_image(valid_image_list, metadata, label):
    valid_samples = list()

    # filter clean images
    if label in ['bear', 'elephant', 'tiger', 'camel', 'gorilla', 'kangaroo', 'zebra']:
        with open(os.path.join('/home/edlab/jylee/RELIABLE/data/clean_imagenet', f'{label}.txt'), 'r') as f:
            clean_image_list = f.read().split('\n')
            clean_image_list = clean_image_list[:-1]

        valid_samples.extend(clean_image_list)

    if label in ['camel', 'giraffe', 'gorilla', 'kangaroo', 'rhino']:
        clean_image_list = os.listdir(f'/home/edlab/jylee/RELIABLE/data/animal/{label}/data/test')
        valid_samples.extend(clean_image_list)

    elif label == 'human':
        with open('/home/data_storage/CelebA/celeba/list_eval_partition.txt', 'r') as f:
            image_list = f.read().split('\n')

        # filter only test set
        image_list = [i.split(' ')[0] for i in image_list[:-1] if i.split(' ')[1] == '2']
        valid_samples.extend(image_list)

        image_list = os.listdir('/home/edlab/jylee/RELIABLE/data/LSP/test')
        valid_samples.extend(image_list)

    if len(valid_samples) == 0:
        raise FileNotFoundError(f'{label} does not have any images')

    final_valid_samples = list()

    for v in valid_samples:
        if v[:15] == 'ILSVRC2012_val_':  # if the image is from imagenet
            final_valid_samples.append((v, int(metadata[valid_image_list[v]])))

        elif label in ['camel', 'gorilla', 'kangaroo']:
            if label == 'camel':
                final_valid_samples.append((v, int(metadata['n02437312'])))
            elif label == 'gorilla':
                final_valid_samples.append((v, int(metadata['n02480855'])))
            elif label == 'kangaroo':
                final_valid_samples.append((v, int(metadata['n01877812'])))

        elif label in ['giraffe', 'rhino', 'human']:
            final_valid_samples.append((v, int(-1)))

    return final_valid_samples


def inverse_normalize(input, pre_fn):
    mean = torch.tensor(pre_fn.mean).unsqueeze(-1).unsqueeze(-1).to(input.device)
    std = torch.tensor(pre_fn.std).unsqueeze(-1).unsqueeze(-1).to(input.device)
    # crop_size = pre_fn.crop_size
    input = input[0] * std + mean
    # input = torchvision.transforms.Resize(crop_size)(input)

    return input


def load_model_preprocessing(model_name):
    assert model_name in ['resnet50', 'convnext_base', 'densenet121', 'efficientnet_b0', 'resnext50_32x4d', 'vgg16', 'wide_resnet50_2'], "model out of range!"

    # load pre-trained model weights
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        pre_fn = ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'convnext_base':
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        pre_fn = ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        pre_fn = DenseNet121_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        pre_fn = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'resnext50_32x4d':
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        pre_fn = ResNeXt50_32X4D_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'vgg16':
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        pre_fn = VGG16_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'wide_resnet50_2':
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        pre_fn = Wide_ResNet50_2_Weights.IMAGENET1K_V1.transforms()
    else:
        raise NotImplementedError

    model = model.cuda()
    model.eval()
    return model, pre_fn


def is_correct(model, image, target):
    output = model(image)
    pred = output.max(1, keepdim=True)[1]
    return True if pred == target.item() else False


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -3, 3)
    return perturbed_image


def save_image_fn(perturbed_data, save_path, image_name, label, epsilon, model_name):
    save_image(perturbed_data, os.path.join(save_path, f"{image_name.split('.')[0]}-{label}-{model_name}-{int(epsilon*100)}.jpg"))


def generate_adversarial_sample(image_path, save_path, label, epsilon, model_name, valid_image_list, metadata, number_of_samples_per_case):
    # filter corresponding samples
    sample_list = filter_image(valid_image_list, metadata, label)
    random.shuffle(sample_list)  # random shuffle test samples

    # load model and preprocessing function
    model, pre_fn = load_model_preprocessing(model_name)

    cnt = 0

    for (image_name, target) in sample_list:
        target = torch.tensor([target]).cuda()
        if image_name[:15] == 'ILSVRC2012_val_':
            image = pre_fn(Image.open(os.path.join(image_path, image_name)).convert('RGB'))
        else:
            try:
                image = pre_fn(Image.open(f'/home/edlab/jylee/RELIABLE/data/animal/{label}/data/test/{image_name}').convert('RGB'))
            except FileNotFoundError:
                try:
                    image = pre_fn(Image.open(f'/home/data_storage/CelebA/celeba/img_align_celeba/{image_name}').convert('RGB'))
                except FileNotFoundError:
                    image = pre_fn(Image.open(f'/home/edlab/jylee/RELIABLE/data/LSP/test/{image_name}').convert('RGB'))

        image = image.unsqueeze(0).cuda()
        image.requires_grad = True

        # forward
        output = model(image)

        if target.item() == -1:
            target = output.argmax(dim=-1)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data

        perturbed_data = fgsm_attack(image, epsilon, data_grad)

        correct_pred = is_correct(model, perturbed_data, target)

        if correct_pred is False:
            save_image_fn(
                inverse_normalize(perturbed_data, pre_fn),
                save_path,
                image_name,
                label,
                epsilon,
                model_name
            )

            cnt += 1

        if cnt == number_of_samples_per_case:
            print(f'Finished Saving {cnt} adversarial images for {model_name} label {label} epsilon {epsilon}')
            break

def generate_samples(model, image_path, save_path, number_of_samples_per_case):
    # change this!
    label_list = ['tiger', 'zebra', 'camel', 'elephant', 'bear', 'kangaroo', 'giraffe', 'rhino', 'human', 'gorilla']
    epsilon_list = [.005, .01, .015, .02, .025, .03]

    metadata = return_metadata()
    valid_image_list = return_image_list()

    if model == 'all':
        model_list = ['resnet50', 'convnext_base', 'densenet121', 'efficientnet_b0', 'resnext50_32x4d', 'vgg16', 'wide_resnet50_2']
    else:
        model_list = [model]

    for model_name in model_list:
        for label in label_list:
            for epsilon in epsilon_list:
                generate_adversarial_sample(
                    image_path=image_path,
                    save_path=save_path,
                    label=label,
                    epsilon=epsilon,
                    model_name=model_name,
                    valid_image_list=valid_image_list,
                    metadata=metadata,
                    number_of_samples_per_case=number_of_samples_per_case,
                )


def check_adversarial_samples(save_path):
    valid_samples = return_image_list()
    valid_samples = {v.split(' ')[0]: v.split(' ')[1][:9] for v in valid_samples}

    metadata = return_metadata()

    # adversarial images
    image_list = os.listdir(save_path)
    # filter images with jpg files
    image_list = [image for image in image_list if image[:-3] == 'jpg']

    target_list = [int(metadata[valid_samples[image]]) for image in image_list]


    correct_prediction_list = []

    for image, target in zip(image_list, target_list):

        _, label, model_name, _ = image.split('-')

        # load model and preprocessing function
        model, pre_fn = load_model_preprocessing(model_name)

        # forward
        target = torch.tensor([target]).cuda()
        image = pre_fn(Image.open(os.path.join(save_path, image)).convert('RGB'))
        image = image.unsqueeze(0).cuda()

        correct_pred = is_correct(model, image, target)

        if correct_pred is True:
            correct_prediction_list.append(image)

    if len(correct_prediction_list) == 0:
        print('All samples are successfully generated!')
    else:
        print(f'Total {len(correct_prediction_list)} samples are incorrectly generated: {correct_prediction_list}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet50', 'convnext_base', 'densenet121', 'efficientnet_b0', 'resnext50_32x4d', 'vgg16', 'wide_resnet50_2', 'all'])
    parser.add_argument('--image_path', type=str, default='./')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--number_of_samples_per_case', type=int, help='number of samples to be generated in one label with one severity')
    parser.add_argument('--gpu_num', type=int, help='GPU number to use')
    parser.add_argument('--sanity_check', action='store_true')
    args = parser.parse_args()

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    if args.sanity_check is True:
        check_adversarial_samples()

    else:
        # generate sample
        generate_samples(args.model, args.image_path, args.save_path, args.number_of_samples_per_case)
        print('Done Making Samples! --> Proceed to Sanity Check')
        # check_adversarial_samples()


if __name__ == '__main__':
    main()