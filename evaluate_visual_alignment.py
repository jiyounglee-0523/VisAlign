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

    with open(args.label_path, 'rb') as f:
        label = pickle.load(f)

    result_df = pd.DataFrame(columns=[
        'model',
        'seed',
        'ood',
        'category1',
        'category2',
        'category3',
        'category4',
        'category5',
        'category6',
        'category7',
        'category8',
    ])

    ood_path = args.save_dir
    seed = args.seed
    model_name = args.model_name
    ood_method = args.ood_method
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

    for image in files:
        if ood_score[image] > 1:
            ood_score[image] = 1
            abstention_rate = (ood_score[image])
        else:
            abstention_rate = (1 - ood_score[image])

        # noramlize remaining probability
        prediction = id_prediction[image]

        gt = label[image]
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
        'category1': np.array(category1_list).mean(),
        'category2': np.array(category2_list).mean(),
        'category3': np.array(category3_list).mean(),
        'category4': np.array(category4_list).mean(),
        'category5': np.array(category5_list).mean(),
        'category6': np.array(category6_list).mean(),
        'category7': np.array(category7_list).mean(),
        'category8': np.array(category8_list).mean(),
    }])], ignore_index=True)

    result_df.to_csv(f'./{model_name}_{ood_method}_{seed}.csv')

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default='./', type=str, help='output directory used in test_main.py')
    parser.add_argument('--test_filenames_path', default='./VisAlign_dataset/open_test_set/label', type=str)
    parser.add_argument('--label_path', default='./VisAlign_dataset/open_test_set/labels.pk')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model_name')
    parser.add_argument('--ood_method')

    args = parser.parse_args()

    evaluate(args)

if __name__ == '__main__':
    main()