import torch
from torch import nn


class GainPlanner(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimensions, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, features):
        return self.network(features) / 100


class SupportCritic(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(134, 32, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.readout = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("class_weights", weights.clone())

    def forward(self, x):
        return self.readout(self.body(x))

    def probabilities(self, x):
        shift = (self.class_weights[0] / self.class_weights[1:]).log()[None, :, None, None]
        rates = (self(x) + shift).sigmoid()
        s = x[:, -1:]
        fp = s * rates[:, 0:1]
        fn = (1 - s) * rates[:, 1:2]
        return torch.cat((1 - fp - fn, fp, fn), 1)
