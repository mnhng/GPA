import copy
import torch
import torch.nn as nn
from .nn_blk import FusionNet, EmbedNet


def get_head(name):
    if name == 'copa':
        return CoPAHead
    elif name == 'gpa':
        return GPAHead
    elif name == 'erm2':
        return LinearZHead

    return LinearHead


def normalize(unnormalized_prob):
    return unnormalized_prob / unnormalized_prob.sum(dim=-1, keepdims=True)


class LinearHead(nn.Linear):
    def __init__(self, dim, nb_Y, nb_Z, **kwargs):
        super().__init__(dim, nb_Y, **kwargs)

    def forward(self, input, **kwargs):
        return super().forward(input)


class LinearZHead(nn.Module):
    def __init__(self, dim, nb_Y, Zs):
        super().__init__()
        self.emb = EmbedNet(Zs)
        self.net = FusionNet(self.emb.dim() + dim, nb_Y, hid_size=100)

    def forward(self, input, Z, **kwargs):
        return self.net(torch.cat([self.emb(Z), input], dim=-1))


class CoPAHead(nn.Module):
    def __init__(self, dim, nb_Y, Zs):
        super().__init__()
        self.emb = EmbedNet(Zs)
        self.net = FusionNet(self.emb.dim() + dim, nb_Y, hid_size=100)

    def _get_ratio(self, input, Z):
        fused = self.net(torch.cat([self.emb(Z), input], dim=-1))
        return nn.functional.log_softmax(fused, dim=-1)

    def forward(self, input, Z, prior='true_prior', no_Z=False, **kwargs):
        if no_Z:
            if prior == 'uniform':
                log_prior = 0
            else:  # NOTE: use P(Y) instead of P(Y|Z)
                kwargs[prior].eval()
                Z_dummy = torch.full_like(Z, torch.nan, dtype=torch.float)
                log_prior = nn.functional.log_softmax(kwargs[prior](Z_dummy), dim=-1)

            nZ = 5
            total = 0
            for candidate in self.emb.sample(nZ):
                Z_vec = candidate.expand(Z.size(0), -1)
                assert Z_vec.numel() == Z.numel(), f'{Z_vec.shape=} {Z.shape=}'
                total += nn.functional.softmax(self._get_ratio(input, Z_vec) + log_prior, dim=-1)

            return (total / nZ).log()  # NOTE: not the same, unable to invert softmax

        kwargs[prior].eval()
        log_prior = nn.functional.log_softmax(kwargs[prior](Z), dim=-1)
        return self._get_ratio(input, Z) + log_prior


class GPAHead(nn.Module):
    def __init__(self, dim, nb_Y, Zs):
        super().__init__()
        self.emb = EmbedNet(Zs, p_knockout=.5)
        self.net = FusionNet(self.emb.dim() + dim, nb_Y, hid_size=100)
        self.ws = nn.Parameter(torch.ones((1, nb_Y), requires_grad=True))
        self.bs = nn.Parameter(torch.zeros((1, nb_Y), requires_grad=True))

    def _get_ratio(self, input, Z):
        fused = self.net(torch.cat([self.emb(Z), input], dim=-1))
        return nn.functional.log_softmax(fused, dim=-1)

    def forward(self, input, Z, prior='true_prior', no_Z=False, **kwargs):
        if no_Z:
            assert not self.training
            bZ = torch.full_like(Z, torch.nan, dtype=torch.float)
        else:
            bZ = Z

        if prior in ('em_prior', 'true_prior'):
            kwargs[prior].eval()
            log_prior = nn.functional.log_softmax(kwargs[prior](bZ), dim=-1)
        elif prior == 'uniform':
            log_prior = 0
        else:
            raise ValueError(f'unsupported {prior=}')

        return (self._get_ratio(input, bZ) + log_prior)*self.ws + self.bs
