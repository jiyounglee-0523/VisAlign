"""Generate a submission file for the VisAlign leaderboard.

Leaderboard: https://huggingface.co/spaces/jiyounglee0523/leaderboard
Dataset:     https://huggingface.co/datasets/jiyounglee0523/VisAlign

The leaderboard expects a JSON file mapping each of the 900 open-test-set
filenames to an 11-dimensional distribution over
[tiger, zebra, camel, giraffe, elephant, rhino, gorilla, bear, kangaroo, human, abstain].

Two ways to create it:

1) From this repo's baseline pipeline outputs (test_main.py results):
    python make_leaderboard_submission.py from-results \
        --save_dir {save_dir} --model_name {model_name} \
        --ood_method {ood_method} --seed {seed} \
        --output my_submission.json

2) From your own model, by implementing predict() (see predictor_template.py):
    python make_leaderboard_submission.py custom \
        --predictor my_predictor.py --output my_submission.json

Then upload the JSON on the "Submit" tab of the leaderboard Space.
"""
import argparse
import importlib.util
import json
import os
import pickle

import numpy as np

CLASSES = ['tiger', 'zebra', 'camel', 'giraffe', 'elephant', 'rhino',
           'gorilla', 'bear', 'kangaroo', 'human', 'abstain']
N_TEST_IMAGES = 900
LEADERBOARD_URL = 'https://huggingface.co/spaces/jiyounglee0523/leaderboard'
DATASET_REPO = 'jiyounglee0523/VisAlign'


def validate_dist(file_name, dist):
    if dist is None:
        return None
    dist = np.asarray(dist, dtype=float).reshape(-1)
    if dist.shape != (11,):
        raise ValueError(f'{file_name}: distribution must have 11 dimensions, got {dist.shape}')
    if (dist < 0).any() or not np.isfinite(dist).all() or dist.sum() <= 0:
        raise ValueError(f'{file_name}: distribution must be non-negative, finite, and sum > 0')
    return (dist / dist.sum()).tolist()


def save_submission(predictions, output_path):
    n_abstain_null = sum(1 for v in predictions.values() if v is None)
    if len(predictions) != N_TEST_IMAGES:
        print(f'Warning: expected {N_TEST_IMAGES} images, got {len(predictions)}. '
              f'The leaderboard will reject submissions with missing images.')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f'Saved {len(predictions)} predictions to {output_path}'
          + (f' ({n_abstain_null} null → scored as abstention)' if n_abstain_null else ''))
    print(f'Upload this file on the "Submit" tab of {LEADERBOARD_URL}')


def from_results(args):
    """Convert test_main.py outputs (id_prediction + oodscore pickles) into a
    submission, combining them exactly as evaluate_visual_alignment.py does."""
    import torch
    from scipy.stats import entropy

    id_path = os.path.join(args.save_dir, f'{args.model_name}-{args.seed}-id_prediction.pk')
    ood_path = os.path.join(args.save_dir, f'{args.model_name}-{args.ood_method}-{args.seed}-oodscore.pk')

    with open(id_path, 'rb') as f:
        id_prediction = pickle.load(f)
    with open(ood_path, 'rb') as f:
        ood_score = pickle.load(f)

    if args.ood_method in ['msp', 'odin', 'mcdropout']:
        ood_score = {k: entropy(v) for k, v in ood_score.items()}
    elif args.ood_method in ['deepensemble']:
        ood_score = {k: np.mean(entropy(v, axis=1)) for k, v in ood_score.items()}

    ood_scores = torch.Tensor(list(ood_score.values()))
    ood_scores_min = ood_scores.min().item()
    ood_scores -= ood_scores.min()
    ood_scores_max = ood_scores.max().item()
    ood_score = {k: ((v - ood_scores_min) / ood_scores_max) for k, v in ood_score.items()}

    predictions = {}
    for image, prediction in id_prediction.items():
        score = float(ood_score[image])
        abstention_rate = 1 if score > 1 else (1 - score)

        prediction = torch.as_tensor(prediction).cpu()
        prediction = prediction * (1 - abstention_rate)
        dist = torch.cat((prediction, torch.Tensor([abstention_rate])))
        dist[dist < 0] = 0
        predictions[image] = validate_dist(image, dist.numpy())

    save_submission(predictions, args.output)


def custom(args):
    """Run a user-provided predict() over the open test set from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit('The custom mode requires the `datasets` library: pip install datasets')
    from tqdm import tqdm

    spec = importlib.util.spec_from_file_location('user_predictor', args.predictor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'predict'):
        raise SystemExit(f'{args.predictor} must define predict(image, file_name) '
                         f'returning 11 numbers (or None to abstain). '
                         f'See predictor_template.py.')
    if hasattr(module, 'setup'):
        module.setup()

    ds = load_dataset(DATASET_REPO, split='test')

    predictions = {}
    for example in tqdm(ds, desc='Predicting'):
        file_name = example['file_name']
        dist = module.predict(example['image'], file_name)
        predictions[file_name] = validate_dist(file_name, dist)

    save_submission(predictions, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_results = subparsers.add_parser(
        'from-results', help="build a submission from this repo's test_main.py outputs")
    p_results.add_argument('--save_dir', default='./', type=str,
                           help='directory used as --save_dir in test_main.py')
    p_results.add_argument('--model_name', required=True, help='model name used in test_main.py')
    p_results.add_argument('--ood_method', required=True,
                           choices=['knn', 'mcdropout', 'mds', 'odin', 'msp', 'tapudd', 'deepensemble'],
                           help='postprocessor name used in test_main.py')
    p_results.add_argument('--seed', type=int, default=45, help='seed used in test_main.py')
    p_results.add_argument('--output', default='submission.json', help='output JSON path')
    p_results.set_defaults(func=from_results)

    p_custom = subparsers.add_parser(
        'custom', help='build a submission by running your own model on the HF open test set')
    p_custom.add_argument('--predictor', required=True,
                          help='path to a python file defining predict(image, file_name)')
    p_custom.add_argument('--output', default='submission.json', help='output JSON path')
    p_custom.set_defaults(func=custom)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
