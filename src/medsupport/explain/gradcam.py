"""Grad-CAM (Selvaraju et al., 2017) implemented with forward/backward hooks on
the last DenseNet feature map. Works for both the image-only model and the
fusion model (whose image branch is the same module)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, forward_fn, class_idx: int, input_size: tuple[int, int]) -> np.ndarray:
        """forward_fn() must run the model and return logits (B, C). Returns a
        (H, W) heat-map in [0, 1] for sample 0."""
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        with torch.enable_grad():
            logits = forward_fn()
            score = logits[0, class_idx]
            score.backward()
        if was_training:
            self.model.train()
        A, G = self.activations[0], self.gradients[0]           # (C, h, w)
        weights = G.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * A).sum(0)).cpu().numpy()
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """img_rgb uint8 (H, W, 3); cam float (H, W) in [0,1]. Returns uint8 RGB overlay."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (alpha * heat + (1 - alpha) * img_rgb).clip(0, 255).astype(np.uint8)
    return out
