import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def copy_to(param_dict, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in param_dict.items()}


def init_best(nb_val_sets):
    return {
        'risk_score': [0]*nb_val_sets,
        'iter': [0]*nb_val_sets,
        'state_dict': [None]*nb_val_sets,
    }


def skim(datasets, ref_size):
    portion =  ref_size / sum(len(d) for d in datasets)
    val_data, train_data = [], []
    for d in datasets:
        v, t = torch.utils.data.random_split(d, [portion, 1-portion], generator=torch.Generator().manual_seed(42))
        val_data.append(v)
        setattr(t, 'E', d.E)
        train_data.append(t)

    return train_data, sum(val_data[1:], val_data[0])


class Batchify():
    def __init__(self, dataset, batch_size):
        self.data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.iter_ = self.data.__iter__()

    def __next__(self):
        try:
            return next(self.iter_)
        except StopIteration:
            self.iter_ = self.data.__iter__()
            return next(self.iter_)
