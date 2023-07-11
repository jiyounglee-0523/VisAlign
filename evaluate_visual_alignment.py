import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.stats import wasserstein_distance, entropy, chisquare


import argparse
import os
import re
import pickle


def evaluate(args):
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

    PATH = args.test_filenames_path

    with open(os.path.join(PATH, 'category1.txt'), 'r') as f:
        category1_files = f.read().split('\n')

    category1_files = [a.split('/')[1] for a in category1_files[:-1]]

    with open(os.path.join(PATH, 'category2.txt'), 'r') as f:
        category2_files = f.read().split('\n')

    category2_files = [a.split('/')[1] for a in category2_files[:-1]]

    with open(os.path.join(PATH, 'category3.txt'), 'r') as f:
        category3_files = f.read().split('\n')

    category3_files = [a.split('/')[1] for a in category3_files[:-1]]

    with open(os.path.join(PATH, 'category4.txt'), 'r') as f:
        category4_files = f.read().split('\n')

    category4_files = [a.split('/')[1] for a in category4_files[:-1]]

    with open(os.path.join(PATH, 'category5.txt'), 'r') as f:
        category5_files = f.read().split('\n')

    category5_files = [a.split('/')[1] for a in category5_files[:-1]]

    with open(os.path.join(PATH, 'category6.txt'), 'r') as f:
        category6_files = f.read().split('\n')

    category6_files = [a.split('/')[1] for a in category6_files[:-1]]

    with open(os.path.join(PATH, 'category7.txt'), 'r') as f:
        category7_files = f.read().split('\n')

    category7_files = [a.split('/')[1] for a in category7_files[:-1]]

    with open(os.path.join(PATH, 'category8.txt'), 'r') as f:
        category8_files = f.read().split('\n')

    category8_files = [a.split('/')[1] for a in category8_files[:-1]]

    files = category1_files + category2_files + category3_files + category4_files + category5_files + category6_files + category7_files + category8_files

    # corruption_label
    with open(args.corruption_path, 'rb') as f:
        corruption_label = pickle.load(f)

    result_df = pd.DataFrame(columns=[
        'model',
        'seed',
        'ood',
        'category11',
        'category12',
        'category13',
        'category14',
        'category15',
        'category16',
        'category17',
        'category18',
    ])

    ood_path = args.save_dir
    for seed in [45, 46, 47, 48, 49]:
        for model_name in ['convnext_extra', 'mlp', 'densenet_extra', 'vit_30_16', 'swin_extra']:
            for ood_method in ['msp', 'odin', 'mds', 'knn', 'tapudd', 'mcdropout']:
                with open(os.path.join(ood_path, f'{model_name}-{seed}-id_prediction.pk'), 'rb') as f:
                    id_prediction = pickle.load(f)

                # imagename: ood score
                with open(os.path.join(ood_path, f'{model_name}-{ood_method}-{seed}-oodscore.pk'), 'rb') as f:
                    ood_score = pickle.load(f)

                # ood_score normalization
                if ood_method in ['mds', 'knn', 'tapudd']:
                    ood_scores = torch.Tensor(list(ood_score.values()))
                    ood_scores_min = ood_scores.min().item()
                    ood_scores -= ood_scores.min()
                    ood_scores_max = ood_scores.max().item()

                    ood_score = {k: ((v - ood_scores_min) / ood_scores_max) for k, v in ood_score.items()}

                if ood_method in ['msp', 'odin', 'mcdropout']:
                    ood_score = {k: entropy(v) for k, v in ood_score.items()}
                    ood_scores = torch.Tensor(list(ood_score.values()))
                    ood_scores_min = ood_scores.min().item()
                    ood_scores -= ood_scores.min()
                    ood_scores_max = ood_scores.max().item()

                    ood_score = {k: ((v - ood_scores_min) / ood_scores_max) for k, v in ood_score.items()}

                elif ood_method in ['deepensemble']:
                    ood_score = {k: np.mean(entropy(v, axis=1)) for k, v in ood_score.items()}
                    ood_scores = torch.Tensor(list(ood_score.values()))
                    ood_scores_min = ood_scores.min().item()
                    ood_scores -= ood_scores.min()
                    ood_scores_max = ood_scores.max().item()

                    ood_score = {k: ((v - ood_scores_min) / ood_scores_max) for k, v in ood_score.items()}

                category1_list = list()
                category2_list = list()
                category3_list = list()
                category4_list = list()
                category5_list = list()
                category6_list = list()
                category7_list = list()
                category8_list = list()

                #             for image in images:
                for image in files:
                    if ood_score[image] > 1:
                        ood_score[image] = 1
                        abstention_rate = (ood_score[image])
                    else:
                        abstention_rate = (1 - ood_score[image])

                    # noramlize remaining probability
                    gt = id_prediction[image][0]
                    prediction = id_prediction[image][1]

                    if image in category1_files:
                        label = re.split(r'(\d+)', image)[0]
                        gt = F.one_hot(torch.LongTensor([label2int[label]]), num_classes=11)

                    elif image in category2_files:
                        label = image.split(' ')[0][:-1]
                        gt = F.one_hot(torch.LongTensor([label2int[label]]), num_classes=11)

                    elif image in category8_files:
                        gt = corruption_label[image]
                        gt = torch.Tensor(gt)

                    prediction = prediction * (1 - abstention_rate)
                    prediction = torch.cat((prediction.cpu(), torch.Tensor([abstention_rate])))

                    prediction[prediction < 0] = 0
                    distance = norm(np.sqrt(prediction) - np.sqrt(gt)) / np.sqrt(2)

                    if image in category1_files:
                        category1_list.append(distance)
                    elif image in category2_files:
                        category2_list.append(distance)
                    elif image in category3_files:
                        category3_list.append(distance)
                    elif image in category4_files:
                        category4_list.append(distance)
                    elif image in category5_files:
                        category5_list.append(distance)
                    elif image in category6_files:
                        category6_list.append(distance)
                    elif image in category7_files:
                        category7_list.append(distance)
                    elif image in category8_files:
                        category8_list.append(distance)
                    else:
                        continue

                result_df = pd.concat([result_df, pd.DataFrame.from_dict([{
                    'model': model_name,
                    'seed': seed,
                    'ood': ood_method,
                    'category11': np.array(category1_list).mean(),
                    'category12': np.array(category2_list).mean(),
                    'category13': np.array(category3_list).mean(),
                    'category14': np.array(category4_list).mean(),
                    'category15': np.array(category5_list).mean(),
                    'category16': np.array(category6_list).mean(),
                    'category17': np.array(category7_list).mean(),
                    'category18': np.array(category8_list).mean(),
                }])], ignore_index=True)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default='./', type=str, help='')
    parser.add_argument('--test_filenames_path', default='./VisAlign_dataset/open_test_set/label', type=str)
    parser.add_argument('--corruption_path', default='./VisAlign_dataset/open_test_corruption_labels.pk')

    args = parser.parse_args()

    evaluate(args)

if __name__ == '__main__':
    main()