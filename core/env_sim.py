import pathlib

import torch
import torch.nn as nn
import numpy as np

from .dataset import QuadInMem
from .io import load_idx
from .prevalence import est_multivariate_g
from .nn_blk import EmbedNet


__all__ = ['SynDia', 'SynTriL', 'SynTriR',
           'CmnistDia', 'CmnistTriL', 'CmnistTriR', 'CmnistDataSplit',
           'SIM_Z_DICT',
           ]

SIM_AKL = 0.01

SIM_Z_DICT = {
    'value': {'one': 0, 'two': 1},
    'color': {'red': 0, 'green': 1},
}


class GNet(nn.Module):
    def __init__(self, nb_Y, Zs, hid_size=100):
        super().__init__()
        # self.emb = EmbedNet(Zs, p_knockout=.5)
        self.emb = EmbedNet(Zs)  # NOTE: knockout training is done in prevalence.py
        self.net = nn.Sequential(
            nn.Linear(self.emb.dim(), hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, nb_Y),
        )

    def forward(self, Z):
        param = next(self.parameters())
        return self.net(self.emb(Z.to(device=param.device, dtype=param.dtype)))


def shuffle_(*arrays):
    idxs = np.random.permutation(len(arrays[0]))
    return [array[idxs] for array in arrays]


def rand_tensor(rng, size):
    return torch.tensor(rng.random(size), dtype=torch.float)


class SynGenerator():
    def _generator_init(self, scramble=None):
        self.mu_y = torch.tensor([-.1, .1])
        self.mu_z = torch.tensor([ -1,  1])
        self.std_y = 0.1
        self.std_z = 0.1
        self.scramble = scramble
        self.Z_maps = [SIM_Z_DICT['value']]

    def _generate(self, Y, Z, rng):
        c1 = torch.where(Y==1, self.mu_y[1], self.mu_y[0]) + torch.tensor(rng.normal(scale=self.std_y, size=Y.shape), dtype=torch.float)
        c2 = torch.where(Z==1, self.mu_z[1], self.mu_z[0]) + torch.tensor(rng.normal(scale=self.std_z, size=Z.shape), dtype=torch.float)

        X = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), 1)
        if self.scramble is not None:
            X = X @ self.scramble

        ids = np.asarray([f'{a:.2f}{b:.2f}' for a, b in zip(c1, c2)])

        return ids, X


class CmnistGenerator():
    def _generator_init(self, ids, images, labels):
        self.ids = ids
        self.images = images.float() / 255.

        indices = torch.arange(len(labels), dtype=int)
        neg_mask = labels < 5
        self.pos = indices[~neg_mask]
        self.neg = indices[neg_mask]
        self.Z_maps = [SIM_Z_DICT['color']]

    def _generate(self, Y, Z, rng):
        indices = torch.where(Y==1, self.pos[:len(Y)], self.neg[:len(Y)])
        images = self.images[indices]
        ids = self.ids[indices]

        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.arange(len(images), dtype=int), Z, :, :] *= 0

        return ids, images


class CmnistDataSplit():
    def __init__(self, root, n_valid, n_test):
        IMG_PATH = pathlib.Path(root)/'train-images-idx3-ubyte.gz'
        LBL_PATH = pathlib.Path(root)/'train-labels-idx1-ubyte.gz'
        # Load MNIST
        self.imgs = torch.tensor(load_idx(IMG_PATH))
        self.lbls = torch.tensor(load_idx(LBL_PATH))
        self.ids = np.asarray([str(i) for i in range(len(self.imgs))])
        self.j = self.imgs.shape[0] - n_test
        self.i = self.j - n_valid
        assert self.i > 0 and self.j > 0

    def shuffle(self):
        imgs, lbls, ids = shuffle_(self.imgs, self.lbls, self.ids)
        train = {'images': imgs[:self.i], 'labels': lbls[:self.i], 'ids': ids[:self.i]}
        valid = {'images': imgs[self.i:self.j], 'labels': lbls[self.i:self.j], 'ids': ids[self.i:self.j]}
        test = {'images': imgs[self.j:], 'labels': lbls[self.j:], 'ids': ids[self.j:]}

        return train, valid, test


#         S
#        / \
#       Y   Z
#        \ /
#         X
class Diamond():
    def __init__(self, beta_y, beta_z, seed, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'
        self.rng = np.random.default_rng(seed=seed)

    def _get_YZ(self, n):
        S = rand_tensor(self.rng, n)
        Y = (self.beta_y*S + (1-self.beta_y)*self.alpha > self.gamma).long()
        Z = (self.beta_z*S + (1-self.beta_z)*rand_tensor(self.rng, n) > .5).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z, self.rng)

        eY, eZ = self._get_YZ(n)
        true_prior = GNet(2, self.Z_maps)
        slack = est_multivariate_g(true_prior, eY, eZ[:, None], akl=SIM_AKL)
        print(f'Param {self.beta_y}: {slack=}')

        return QuadInMem(X, Y, 1+Z[:, None], self.label, ids, true_prior)


#       Y-->Z
#        \ /
#         X
class TriagL():
    def __init__(self, beta_y, beta_z, seed, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'
        self.rng = np.random.default_rng(seed=seed)

    def _get_YZ(self, n):
        Y = (self.beta_y*rand_tensor(self.rng, n) + (1-self.beta_y)*self.alpha > self.gamma).long()
        Z = (self.beta_z*Y/2 + (1-self.beta_z/2)*rand_tensor(self.rng, n) > .5).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z, self.rng)

        eY, eZ = self._get_YZ(n)
        true_prior = GNet(2, self.Z_maps)
        slack = est_multivariate_g(true_prior, eY, eZ[:, None], akl=SIM_AKL)
        print(f'Param {self.beta_y}: {slack=}')

        return QuadInMem(X, Y, 1+Z[:, None], self.label, ids, true_prior)


#       Y<--Z
#        \ /
#         X
class TriagR():
    def __init__(self, beta_y, beta_z, seed, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'
        self.rng = np.random.default_rng(seed=seed)

    def _get_YZ(self, n):
        Z = (rand_tensor(self.rng, n) > .5).long()
        Y = (self.beta_y*(Z+rand_tensor(self.rng, n))/2 + (1-self.beta_y)*self.alpha > self.gamma).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z, self.rng)

        eY, eZ = self._get_YZ(n)
        true_prior = GNet(2, self.Z_maps)
        slack = est_multivariate_g(true_prior, eY, eZ[:, None], akl=SIM_AKL)
        print(f'Param {self.beta_y}: {slack=}')

        return QuadInMem(X, Y, 1+Z[:, None], self.label, ids, true_prior)


class SynDia(Diamond, SynGenerator):
    pass


class SynTriL(TriagL, SynGenerator):
    pass


class SynTriR(TriagR, SynGenerator):
    pass


class CmnistDia(Diamond, CmnistGenerator):
    pass


class CmnistTriL(TriagL, CmnistGenerator):
    pass


class CmnistTriR(TriagR, CmnistGenerator):
    pass
