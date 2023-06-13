import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from scipy import linalg
import sklearn.covariance
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from typing import Any
from tqdm import tqdm

from .base_postprocessor import BasePostProcessor

class TAPUDDPostProcessor(BasePostProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.postprocessor_args = config.postprocessor['postprocessor_args']
        self.feature_type_list = self.postprocessor_args['feature_type_list']
        self.magnitude = self.postprocessor_args['noise']

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.feature_mean, self.feature_prec = get_MDS_stat(net, id_loader_dict['train'], self.feature_type_list)


    def postprocess(self, net:nn.Module, data: Any):

        for k_idx in range(len(self.feature_mean)):
            score = compute_Mahalanobis_score(
                net,
                Variable(data, requires_grad=True),
                k_idx+1,
                self.feature_mean[k_idx],
                self.feature_prec[k_idx],
                self.feature_type_list,
                self.magnitude
            )

            if k_idx == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)

        conf = score_list.mean(axis=1)
        return conf



@torch.no_grad()
def get_MDS_stat(model, train_loader, feature_type_list):
    """
    Compute sample mean and precision (inverse of covariance)
    return:
        sample_class_mean = list of class mean
        precision = list of precisions
        transform_matrix_list = list of transform matrix
    """

    model.eval()
    num_layer = len(feature_type_list)
    feature_all = [None for _ in range(num_layer)]

    total_feature_list = list()
    total_precision_list = list()

    for image, label in tqdm(train_loader, desc='Compute mean/std'):
        image = image.cuda()

        feature_list = [model(image)]

        for layer_idx in range(num_layer):
            feature_type = feature_type_list[layer_idx]
            feature_processed = process_feature_type(feature_list[layer_idx], feature_type)

            if isinstance(feature_all[layer_idx], type(None)):
                feature_all[layer_idx] = tensor2list(feature_processed)
            else:
                feature_all[layer_idx].extend(tensor2list(feature_processed))


    for num_classes in range(1, 11):
        gmm = GaussianMixture(n_components=num_classes, random_state=42).fit(feature_all[0])
        feature_mean_list = gmm.means_
        precision_list = gmm.precisions_

        feature_mean_list = [torch.Tensor(i).cuda() for i in feature_mean_list]
        precision_list = [torch.Tensor(p).cuda() for p in precision_list]

        total_feature_list.append(feature_mean_list)
        total_precision_list.append(precision_list)

    return total_feature_list, total_precision_list




def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        feature_temp = feature_temp.view([feature_temp.size(0), -1])
    elif feature_type == 'stat':
        feature_temp = get_torch_feature_stat(feature_temp)
    elif feature_type == 'mean':
        feature_temp = get_torch_feature_stat(feature_temp, only_mean=True)
    else:
        raise ValueError('Unknown feature type')
    return feature_temp

def tensor2list(x):
    return x.data.cpu().tolist()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)

    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        feature_stat = feature_mean
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def get_Mahalanobis_scores(model, test_loader, num_classes, sample_mean,
                           precision, transform_matrix, layer_index,
                           feature_type_list, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return:
        Mahalanobis score from layer index
    '''
    model.eval()
    Mahalnobis = []

    for batch in tqdm(test_loader):
        data = batch['data'].cuda()
        data = Variable(data, requires_grad=True)
        noise_gaussian_score = compute_Mahalanobis_score(
            model, data, num_classes, sample_mean, precision, transform_matrix,
            layer_index, feature_type_list, magnitude
        )
        Mahalnobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalnobis



def compute_Mahalanobis_score(model, data, num_classes, sample_mean, precision,
                              feature_type_list, magnitude, return_pred=False):
    # extract features
    layer_index = 0
    out_features = [model(data)]
    out_features = process_feature_type(out_features[layer_index], feature_type_list[layer_index])

    # compute Mahalanobis score
    gaussian_score = 0

    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()

        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

    # Input processing
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean.unsqueeze(-1))
    pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    gradient.index_copy_(
        1,
        torch.LongTensor([0]).cuda(),
        gradient.index_select(1, torch.LongTensor([0]).cuda()) / 0.5
    )

    gradient.index_copy_(
        1,
        torch.LongTensor([1]).cuda(),
        gradient.index_select(1, torch.LongTensor([1]).cuda()) / 0.5
    )

    gradient.index_copy_(
        1,
        torch.LongTensor([2]).cuda(),
        gradient.index_select(1, torch.LongTensor([2]).cuda()) / 0.5
    )

    tempInputs = torch.add(data.data, gradient, alpha=-magnitude)

    with torch.no_grad():
        noise_out_features = [model(Variable(tempInputs))]
        noise_out_features = process_feature_type(noise_out_features[layer_index], feature_type_list[layer_index])

    noise_gaussian_score = 0

    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()

        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat(
                (noise_gaussian_score, term_gau.view(-1, 1)), 1
            )

    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

    if return_pred:
        return sample_pred, noise_gaussian_score
    else:
        return noise_gaussian_score
