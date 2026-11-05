import math
import importlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_ENCODER = "dinov3_vits16"


class FrozenDINO(nn.Module):
    def __init__(self, repo, weights, name=DEFAULT_ENCODER):
        super().__init__()
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        if name.startswith("dinov3_"):
            if repo:
                sys.path.insert(0, str(Path(repo).resolve()))
            factory = getattr(importlib.import_module("dinov3.hub.backbones"), name)
            self.backbone = factory(weights=str(Path(weights).resolve()))
        else:
            self.backbone = torch.hub.load(str(repo), name, source="local", weights=str(weights))
        self.name = name
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.channels = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        depth = len(self.backbone.blocks)
        self.layers = [math.ceil(depth * fraction) - 1 for fraction in (0.25, 0.5, 0.75, 1.0)]

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image):
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height % self.patch_size, -width % self.patch_size)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                image, n=self.layers, reshape=True, norm=True
            )
        return (features, image.shape[-2:])


def conv_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, channels),
        nn.GELU(),
    )
