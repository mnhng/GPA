import torch
import torch.nn as nn


class CatKnockout(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, Z0=None, **kwargs):
        super().__init__(num_embeddings+1, embedding_dim, **kwargs)
        self.Z0 = num_embeddings if Z0 is None else Z0

    def forward(self, in_):
        return super().forward(torch.where(torch.isnan(in_), self.Z0, in_).int())


class ContKnockoutBounded(nn.Module):
    def __init__(self, amin=None, amax=None):
        super().__init__()
        if amin is None:
            assert amax is None
            self.amin, self.amax = 0, 1
        else:  # map from [amin, amax] to [0, 1]
            assert amax > amin, f'{amin=} {amax=}'
            self.amin, self.amax  = amin, amax

    def forward(self, in_):
        return torch.where(torch.isnan(in_), -1, (in_ - self.amin) / (self.amax - self.amin))


class EmbedNet(nn.Module):
    def __init__(self, Z_maps, p_knockout=0, embed_size=5):
        super().__init__()
        self.mod_list = nn.ModuleList()
        self.out_size = 0
        self.p = 1 - p_knockout
        for z_map in Z_maps:
            if len(z_map) == 1:  # a float variable
                assert 'float' in z_map, z_map
                self.mod_list.append(ContKnockoutBounded(*z_map['float']))
                self.out_size += 1
            else: # a categorical variable
                self.mod_list.append(CatKnockout(max(z_map.values())+1, embed_size))
                self.out_size += embed_size

    def dim(self):
        return self.out_size

    def sample(self, n):
        dev = next(self.parameters()).device
        out = []
        for module in self.mod_list:
            if isinstance(module, CatKnockout):
                out.append(torch.randint(0, module.num_embeddings-1, size=(n, 1)))
            elif isinstance(module, ContKnockoutBounded):
                out.append(torch.FloatTensor(n, 1).uniform_(module.amin, module.amax))
            else:
                raise ValueError()

        return torch.cat(out, dim=-1).to(device=dev, dtype=torch.float)

    def forward(self, Z_vec, **kwargs):
        if Z_vec.dim() == 1:
            Z_vec = Z_vec[:, None]

        if self.training and self.p < 1:
            retained_mask = torch.empty_like(Z_vec).bernoulli_(p=self.p).bool()
            Z_vec = torch.where(retained_mask, Z_vec, torch.nan)

        assert len(self.mod_list) == Z_vec.shape[1]
        out = []
        for i, module in enumerate(self.mod_list):
            if isinstance(module, CatKnockout):
                out.append(module(Z_vec[:, i]))
            elif isinstance(module, ContKnockoutBounded):
                out.append(module(Z_vec[:, i])[:, None])
            else:
                raise ValueError()

        return torch.cat(out, dim=-1)


class FusionNet(nn.Sequential):
    def __init__(self, in_size, out_size, hid_size):
        super().__init__(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size),
        )

    def forward(self, in_):
        param = next(self.parameters())
        return super().forward(in_.to(device=param.device, dtype=param.dtype))
