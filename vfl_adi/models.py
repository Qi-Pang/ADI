from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class HeteroLR(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, num_logits: int = 1):
        super().__init__()
        self.model_name = "HeteroLR"
        self.local_a = nn.Linear(dim_a, num_logits)
        self.local_b = nn.Linear(dim_b, num_logits)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        a = self.local_a(x_a.flatten(1))
        b = self.local_b(x_b.flatten(1))
        return a + b


def _mnist_conv(ch: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, ch, kernel_size=3), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, kernel_size=3), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(ch, ch, kernel_size=3), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
    )


class SplitNNMnist(nn.Module):
    def __init__(self, num_classes: int = 10, ch: int = 8):
        super().__init__()
        self.model_name = "SplitNNMnist"
        self.local_a = _mnist_conv(ch)
        self.local_b = _mnist_conv(ch)
        emb = ch * 10 * 3
        self.top = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2 * emb, num_classes))

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        a = self.local_a(x_a).flatten(1)
        b = self.local_b(x_b).flatten(1)
        return self.top(torch.cat((a, b), dim=1))


class SplitNNNusWide(nn.Module):
    def __init__(self, dim_a: int = 634, dim_b: int = 1000, num_classes: int = 10):
        super().__init__()
        self.model_name = "SplitNNNusWide"
        self.local_a = nn.Sequential(
            nn.Linear(dim_a, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
        )
        self.local_b = nn.Sequential(
            nn.Linear(dim_b, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
        )
        self.top = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        a = self.local_a(x_a.flatten(1))
        b = self.local_b(x_b.flatten(1))
        return self.top(torch.cat((a, b), dim=1))


class SplitNNCifar(nn.Module):
    def __init__(self, dim_a: int = 8192, dim_b: int = 8192, num_classes: int = 10):
        super().__init__()
        self.model_name = "SplitNNCifar"
        self.local_a = nn.Sequential(nn.Linear(dim_a, 256), nn.ReLU(inplace=True))
        self.local_b = nn.Sequential(nn.Linear(dim_b, 256), nn.ReLU(inplace=True))
        self.top = nn.Sequential(
            nn.Dropout(), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        a = self.local_a(x_a.flatten(1))
        b = self.local_b(x_b.flatten(1))
        return self.top(torch.cat((a, b), dim=1))


def vgg16_feature_extractor() -> nn.Module:
    net = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
    for p in net.parameters():
        p.requires_grad = False
    return net.eval()
