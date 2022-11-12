import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import QuadInMem
from .prevalence import est_multivariate_g
from .nn_blk import EmbedNet


ISIC_AKL = 0.1

ISIC_Y = {  # NOTE: no mapping for NaN
    'benign': 0,
    'malignant': 1
}

ISIC_Z_DICT = {
    'anatom_site_general': {
        'anterior torso': 0,
        'head/neck': 1,
        'lateral torso': 2,
        'lower extremity': 3,
        'oral/genital': 4,
        'palms/soles': 5,
        'posterior torso': 6,
        'upper extremity': 7,
        float('nan'): 8,
    },
    'sex': {'female': 0, 'male': 1, float('nan'): 2},
    # image_type
    'diagnosis_confirm_type': {
        'confocal microscopy with consensus dermoscopy': 0,
        'histopathology': 1,
        'serial imaging showing no change': 2,
        'single image expert consensus': 3,
        float('nan'): 4,
    },
    'dermoscopic_type': {
        'contact non-polarized': 0,
        'contact polarized': 1,
        'non-contact polarized': 2,
        float('nan'): 3,
    },
    'image_type': {
        'dermoscopic': 0,
        'overview': 1,
        float('nan'): 2,
    },
    'age_approx': {'float': [0, 150]},
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


def ISIC_site(img_dir, site, zlabels, fit_cond_prev=False):
    cache = torch.load(img_dir/f'{site}.pt')
    df = pd.read_csv('datasets/ISIC_pooled/ISIC_consolidated.csv', low_memory=False)
    df = df[df['site'] == site]
    assert len(df) == len(cache), f'{len(df)=} {len(cache)=}'

    mask = df['benign_malignant'].isin(ISIC_Y)
    X = cache[np.arange(len(df))[mask]]
    df = df[mask]

    Y = torch.as_tensor(df['benign_malignant'].map(ISIC_Y).to_numpy()).long()

    Z = []
    for zl in zlabels:
        if zl not in df.columns:
            Z.append(np.full(len(df), np.nan))
        elif len(ISIC_Z_DICT[zl]) == 1 and 'float' in ISIC_Z_DICT[zl]:
            Z.append(df[zl].to_numpy())
        else:
            Z.append(df[zl].map(ISIC_Z_DICT[zl]).to_numpy())

    Z = torch.cat([torch.tensor(col[:, None]) for col in Z], dim=-1)

    if fit_cond_prev:
        cond_prev = GNet(len(ISIC_Y), [ISIC_Z_DICT[zl] for zl in zlabels])
        accuracy = est_multivariate_g(cond_prev, Y, Z, akl=ISIC_AKL)
        print(f'Estimated prior at {site=} {accuracy=}')
    else:
        cond_prev = None

    return QuadInMem(X, Y, Z, site, df.isic_id.to_numpy(), cond_prev)
