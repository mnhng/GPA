import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import QuadInMem
from .prevalence import est_multivariate_g
from .nn_blk import EmbedNet


CXR_AKL = 1

CXR_Z_DICT = {
    'Age': {'float': [0, 150]},
    'Sex': {'F': 0, 'M': 1, 'U': 2, float('nan'): 3},
    'AP/PA': {'AP': 0, 'PA': 1, 'LL': 2, 'L': 3},
}


class GNet(nn.Module):
    def __init__(self, nb_Y, Zs, hid_size=100):
        super().__init__()
        self.emb = EmbedNet(Zs)
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


def CXR_site(root_dir, site, zlabels, fit_cond_prev=False):
    if site == 'CXR8':
        X = torch.load(root_dir/'cxr8.pt')
        df = pd.read_csv(root_dir/'cxr8.csv')
    elif site == 'PadChest':
        X = torch.load(root_dir/'padchest.pt')
        df = pd.read_csv(root_dir/'padchest.csv')
    elif site == 'CheXpert':
        X = torch.load(root_dir/'cxpert.pt')
        df = pd.read_csv(root_dir/'cxpert.csv')
    elif site == 'mimic':
        X = torch.load(root_dir/'mimic_1_per_subj.pt')
        df = pd.read_csv(root_dir/'mimic_1_per_subj.csv')
    elif site == 'vindr':
        X = torch.load(root_dir/'vindr.pt')
        df = pd.read_csv(root_dir/'vindr.csv')
    else:
        raise ValueError(site)

    assert len(df) == len(X), f'{len(df)=} {len(X)=}'

    Y = torch.as_tensor(df['Pneumonia'].to_numpy()).long()

    Z = []
    for zl in zlabels:
        if zl not in df.columns:
            Z.append(np.full(len(df), np.nan))
        elif len(CXR_Z_DICT[zl]) == 1 and 'float' in CXR_Z_DICT[zl]:
            Z.append(df[zl].to_numpy())
        else:
            Z.append(df[zl].map(CXR_Z_DICT[zl]).to_numpy())

    Z = torch.cat([torch.tensor(col[:, None]) for col in Z], dim=-1)

    if fit_cond_prev:
        cond_prev = GNet(len(Y.unique()), [CXR_Z_DICT[zl] for zl in zlabels])
        accuracy = est_multivariate_g(cond_prev, Y, Z, akl=CXR_AKL)
        print(f'Estimated prior at {site=} {accuracy=}')
    else:
        cond_prev = None

    return QuadInMem(X.expand(-1, 3, -1, -1), Y, Z, site, df.Path.to_numpy(), cond_prev)
