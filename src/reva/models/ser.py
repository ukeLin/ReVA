import torch
from torch import nn
from torch.nn import functional as F
from ..geometry import roi_sample


class CropRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Conv2d(128, 16, 1)
        self.key = nn.Conv2d(128, 16, 1)
        self.mix = nn.Sequential(
            nn.Conv2d(260, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 8, 3, padding=1), nn.GELU()
        )
        self.fine = nn.Sequential(nn.Conv2d(73, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    def response(self, view, prior, global_core, base_core, rgb_core, rel):
        core = roi_sample(view, rel, 14)
        q = F.normalize(self.query(core), dim=1)
        keys = self.key(view)
        weights = torch.cat(((prior - 0.7).clamp_min(0), (0.3 - prior).clamp_min(0)), 1)
        refs = torch.einsum("bchw,bkhw->bkc", keys, weights) / (
            weights.sum((2, 3))[:, :, None] + 1e-05
        )
        refs = F.normalize(refs, dim=-1)
        match = torch.einsum("bchw,bkc->bkhw", q, refs)
        certainty = (weights.sum((2, 3)) > 0.0001).float()[:, :, None, None].expand(-1, -1, 14, 14)
        fused = self.mix(torch.cat((global_core, core, match, certainty), 1))
        fine = torch.cat(
            (
                F.interpolate(fused, (224, 224), mode="bilinear", align_corners=False),
                self.rgb(rgb_core),
                base_core.sigmoid(),
            ),
            1,
        )
        return self.fine(fine)

    def forward(self, view, prior, global_core, base_core, rgb_core, rel, old_view, old_rgb):
        observed = self.response(view, prior, global_core, base_core, rgb_core, rel)
        previous = self.response(old_view, prior, global_core, base_core, old_rgb, rel)
        return observed - previous
