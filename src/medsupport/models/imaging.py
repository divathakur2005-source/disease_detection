"""Imaging branch: DenseNet-121 (ImageNet-pretrained) feature extractor.

Returns both the spatial feature map (for cross-modal attention and Grad-CAM)
and a pooled global vector (for the image-only baseline head).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, n_classes: int | None = None, pretrained: bool = True, dim: int = 256,
                 freeze_until: str | None = None):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        self.features = net.features                     # (B, 1024, 7, 7) for 224x224
        self.feat_channels = 1024
        self.proj = nn.Sequential(nn.Conv2d(self.feat_channels, dim, kernel_size=1), nn.ReLU(inplace=True))
        self.dim = dim
        self.head = nn.Linear(self.feat_channels, n_classes) if n_classes else None
        if freeze_until:
            self._freeze(freeze_until)

    def _freeze(self, until: str) -> None:
        """Freeze layers up to (and excluding) the named child of `features`."""
        for name, child in self.features.named_children():
            if name == until:
                break
            for p in child.parameters():
                p.requires_grad = False

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.features(x), inplace=False)   # (B, 1024, H', W')

    def tokens(self, x: torch.Tensor):
        """Returns (tokens (B, H'W', dim), pooled (B, 1024), fmap (B, 1024, H', W'))."""
        fmap = self.feature_map(x)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        t = self.proj(fmap).flatten(2).transpose(1, 2)
        return t, pooled, fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.feature_map(x)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        if self.head is None:
            return pooled
        return self.head(pooled)
