import torch
import torchvision


class QuadInMem(torch.utils.data.Dataset):
    def __init__(self, X, Y, Z, label, ids, true_prior):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.E = label
        self.ids = ids
        self.true_prior = true_prior
        self.em_prior = None

    def __getitem__(self, index):
        return {
                'id': self.ids[index],
                'X': self.X[index], 'Y': self.Y[index], 'Z': self.Z[index],
                }

    def __len__(self):
        return len(self.X)


class QuadOnDisk(torch.utils.data.Dataset):
    def __init__(self, X_path, Y, Z, YE, YZE, label, ids):
        self.X_path = X_path
        self.Y = Y
        self.Z = Z
        self.YE = YE
        self.YZE = YZE
        self.ids = ids
        self.E = label
        self.aug_fn = torchvision.transforms.RandomCrop(224)

    def __getitem__(self, index):
        img = torchvision.io.read_image(self.X_path[index]).float() / 255.
        return {
                'id': self.ids[index],
                'X': self.aug_fn(img), 'Y': self.Y[index], 'Z': self.Z[index],
                'YE': self.YE[index], 'YZE': self.YZE[index],
                }

    def __len__(self):
        return len(self.X_path)
