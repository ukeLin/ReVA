import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import FrozenDINO, conv_block


def make_components(feature, fraction, delta, mode):
    if mode != "single":
        raise ValueError("ReVA uses shared semantic values")
    value = feature + delta
    return (value, value)


def patch_fraction(mask, padded_size, patch_size=16):
    dh, dw = (padded_size[0] - mask.shape[-2], padded_size[1] - mask.shape[-1])
    if min(dh, dw) < 0:
        raise ValueError("Target is larger than padded encoder input")
    mask = F.pad(mask.float(), (0, dw, 0, dh), mode="replicate")
    return F.avg_pool2d(mask, patch_size, stride=patch_size)


class MixedTokenDecoder(nn.Module):
    def __init__(self, channels, width=128, mode="single", patch_size=16):
        super().__init__()
        if patch_size != 16:
            raise ValueError("First implementation requires a 16-pixel patch")
        self.mode, self.width, self.subpixels = (mode, width, 4)
        self.projections = nn.ModuleList([nn.Conv2d(channels, width, 1) for _ in range(4)])
        self.fusion = nn.Sequential(nn.Conv2d(width * 4, width, 1), conv_block(width))
        self.fraction_head = nn.Conv2d(width, 1, 1)
        self.delta_head = nn.Conv2d(width, width, 3, padding=1)
        self.guidance = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.query = nn.Conv2d(width + 32, 32, 1)
        self.key = nn.Conv2d(width, 32, 1)
        self.role_key = nn.Parameter(torch.zeros(2, 32))
        self.refine = conv_block(width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.delta_head.weight, std=0.001)
        nn.init.zeros_(self.delta_head.bias)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        uy, ux = ((yy.flatten() + 0.5) / 4 - 0.5, (xx.flatten() + 0.5) / 4 - 0.5)
        ny, nx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij")
        distance = (uy[:, None] - ny.flatten()[None]).square() + (
            ux[:, None] - nx.flatten()[None]
        ).square()
        self.register_buffer("position_bias", (-2 * distance).repeat(1, 2), persistent=False)

    @staticmethod
    def neighbors(value):
        b, c, h, w = value.shape
        return F.unfold(value, 3, padding=1).reshape(b, c, 9, h * w).permute(0, 3, 2, 1)

    def queries(self, value, h, w):
        b, c = value.shape[:2]
        return value.reshape(b, c, h, 4, w, 4).permute(0, 2, 4, 3, 5, 1).reshape(b, h * w, 16, c)

    def forward(self, features, padded_size, output_size, image):
        state = self.fusion(torch.cat([p(f) for p, f in zip(self.projections, features)], dim=1))
        b, _, h, w = state.shape
        fraction_logits = self.fraction_head(state)
        fraction = fraction_logits.sigmoid()
        delta = 2 * self.delta_head(state).tanh()
        fg, bg = make_components(state, fraction, delta, self.mode)
        padded = F.pad(
            image,
            (0, padded_size[1] - image.shape[-1], 0, padded_size[0] - image.shape[-2]),
            mode="replicate",
        )
        guidance = self.guidance(padded)
        fine_state = F.interpolate(state, size=(h * 4, w * 4), mode="bilinear", align_corners=False)
        query = self.queries(self.query(torch.cat((fine_state, guidance), dim=1)), h, w)
        keys = torch.cat(
            (
                self.neighbors(self.key(fg)) + self.role_key[0],
                self.neighbors(self.key(bg)) + self.role_key[1],
            ),
            dim=2,
        )
        values = torch.cat((self.neighbors(fg), self.neighbors(bg)), dim=2)
        prior = torch.cat(
            (self.neighbors(0.02 + 0.96 * fraction), self.neighbors(0.02 + 0.96 * (1 - fraction))),
            dim=2,
        ).squeeze(-1)
        valid = self.neighbors(torch.ones_like(fraction)).squeeze(-1).bool().repeat(1, 1, 2)
        score = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        score = (
            score.float() + self.position_bias + prior.float().clamp_min(1e-06).log().unsqueeze(-2)
        )
        score = score.masked_fill(~valid.unsqueeze(-2), -10000.0)
        weight = score.softmax(-1).to(values.dtype)
        assigned = torch.matmul(weight, values)
        dense = (
            assigned.reshape(b, h, w, 4, 4, self.width)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, self.width, h * 4, w * 4)
        )
        dense = self.refine(dense)
        quarter_logits = self.head(dense) + F.interpolate(
            fraction_logits, size=dense.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = F.interpolate(
            quarter_logits, size=padded_size, mode="bilinear", align_corners=False
        )
        logits = logits[..., : output_size[0], : output_size[1]]
        return {
            "logits": logits,
            "fraction_logits": fraction_logits,
            "fraction": fraction,
            "delta": delta,
            "dense_state": state,
            "refined_state": dense,
            "component_fg": fg,
            "component_bg": bg,
            "padded_size": padded_size,
            "role_fg_logits": self.head(fg),
            "role_bg_logits": self.head(bg),
        }


class MixedTokenSegmenter(nn.Module):
    def __init__(self, repo, weights, encoder="dinov3_vits16", mode="single", width=128):
        super().__init__()
        self.encoder = FrozenDINO(repo, weights, encoder)
        self.decoder = MixedTokenDecoder(
            self.encoder.channels, width, mode, self.encoder.patch_size
        )

    def forward(self, image):
        features, padded_size = self.encoder(image)
        return self.decoder(features, padded_size, image.shape[-2:], image)


def component_loss(result, masks, patch_size=16):
    target = patch_fraction(masks, result["padded_size"], patch_size)
    occupancy = F.binary_cross_entropy_with_logits(result["fraction_logits"].float(), target)
    predicted_fraction = patch_fraction(
        result["logits"].float().sigmoid(), result["padded_size"], patch_size
    )
    consistency = (predicted_fraction - result["fraction"].float()).square().mean()
    regularization = result["delta"].float().square().mean()
    role = (
        target * F.softplus(-result["role_fg_logits"].float())
        + (1 - target) * F.softplus(result["role_bg_logits"].float())
    ).mean()
    return (occupancy, consistency, regularization, role)
