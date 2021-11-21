import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import random
import pickle
import datetime
import os.path
import torch.nn.functional as F
from copy import deepcopy

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def cluster_f1(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    precision = w / np.sum(w[row_ind, :], axis = 1, keepdims=True)
    recall = w / np.sum(w[:, col_ind], axis = 0, keepdims=True)
    f1 = 2 * precision * recall / (precision + recall)
    f1_max = np.nanmax(f1, axis=0)
    cls_weight = np.sum(w[:, col_ind], axis = 0)
    cls_weight = cls_weight / np.sum(cls_weight)
    w_f1 = np.sum(cls_weight * f1_max)

    return w_f1

def prepare_save_dir(args, filename):
    """ Create saving directory."""
    runner_name = os.path.basename(filename).split(".")[0]
    model_dir = 'data/experiments/{}/{}/'.format(runner_name, args.name)
    args.savedir = model_dir
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    return args


def write_txt(args, string):
    """ Write the string in a text file."""
    with open(args.savedir + 'out.txt', 'a') as f:
        f.write(string + " \n")


def create_logger(args, metrics):
    """ Create a logger."""
    args.logger = {}
    args.logger['time_start'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    i = 0
    while os.path.isfile(args.savedir + str(i) + '.pkl'):
        i += 1
    args.logger['pkl_path'] = args.savedir + str(i) + '.pkl'
    args.logger['path'] = args.savedir + str(i)
    for metric in metrics:
        args.logger[metric] = []
    return args


def save_logger(args):
    """ Save the logger."""
    with open(args.logger['pkl_path'], "wb") as output_file:
        pickle.dump(vars(args), output_file)

def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)