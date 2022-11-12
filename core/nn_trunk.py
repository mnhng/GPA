import torch.nn as nn
import torchvision


class ResNetWrapper(nn.Module):
    def __init__(self, hid, **kwargs):
        super().__init__()
        featurizer = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        setattr(featurizer, 'fc', nn.Linear(2048, hid))
        self.featurizer = featurizer

    def forward(self, X, **kwargs):
        return self.featurizer(X)


class DenseNetWrapper(nn.Module):
    def __init__(self, hid, **kwargs):
        super().__init__()
        featurizer = torchvision.models.densenet121('IMAGENET1K_V1')
        setattr(featurizer, 'classifier', nn.Linear(1024, hid))
        self.featurizer = featurizer

    def forward(self, X, **kwargs):
        return self.featurizer(X)


class EfficientnetWrapper(nn.Module):
    def __init__(self, hid, **kwargs):
        super().__init__()
        featurizer = torchvision.models.efficientnet_v2_s('IMAGENET1K_V1')
        setattr(featurizer, 'classifier', nn.Linear(1280, hid))
        self.featurizer = featurizer

    def forward(self, X, **kwargs):
        return self.featurizer(X)


class GrayCNN(nn.Sequential):
    def __init__(self, hid, **kwargs):
        super().__init__(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(64, hid),
        )

    def forward(self, X, **kwargs):
        return super().forward(X.sum(dim=1, keepdim=True))


class SmallCNN(nn.Sequential):
    def __init__(self, hid, **kwargs):
        super().__init__(
            nn.Conv2d(2, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(64, hid),
        )

    def forward(self, X, **kwargs):
        return super().forward(X)


def get_trunk(name):
    if name == 'r50':
        return ResNetWrapper
    if name == 'd121':
        return DenseNetWrapper
    if name == 'e2s':
        return EfficientnetWrapper

    if name == 'ora':
        return GrayCNN
    if name == 'cnn':
        return SmallCNN

    raise ValueError(name)
