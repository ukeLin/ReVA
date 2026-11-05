import numpy as np
import torch
from torch.nn import functional as F
from .data import MEAN, STD
from .geometry import relative, roi_sample, paste_roi
from .geometry import geometry as geometry
from .models.ovp import GainPlanner as GainPlanner, SupportCritic as SupportCritic
from .models.ser import CropRefiner as CropRefiner


def image_tensor(image):
    value = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)
    return (value - MEAN) / STD


def policy_features(packet, device):
    state = packet["state"][None].float().to(device)
    boxes = packet["boxes"].to(device)
    local = roi_sample(state.expand(9, -1, -1, -1), boxes, 14)
    probability = packet["logits"][None].float().to(device).sigmoid()
    p = roi_sample(probability.expand(9, -1, -1, -1), boxes, 28).clamp(1e-06, 1 - 1e-06)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean((1, 2, 3))
    features = torch.cat(
        (
            local.mean((2, 3)),
            local.std((2, 3)),
            state.mean((2, 3)).expand(9, -1),
            boxes,
            p.mean((2, 3)),
            p.std((2, 3)),
            entropy[:, None],
            (p > 0.5).float().mean((2, 3)),
        ),
        1,
    )
    return (features, entropy)


def prepare_inputs(packet, ids, device):
    ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    boxes = packet["boxes"].to(device)[ids // 2]
    views = packet["views"].to(device)[ids]
    rel = relative(boxes, views)
    logits = packet["logits"][None].float().to(device)
    state = packet["state"][None].float().to(device)
    global_core = roi_sample(state.expand(len(ids), -1, -1, -1), boxes, 14)
    base_core = roi_sample(logits.expand(len(ids), -1, -1, -1), boxes, 224)
    prior = roi_sample(logits.sigmoid().expand(len(ids), -1, -1, -1), views, 14)
    view = packet["view_state"].to(device)[ids].float()
    rgb = packet["view_rgb"].to(device)[ids].float() / 255
    rgb = (rgb - MEAN.to(device)) / STD.to(device)
    rgb_core = roi_sample(rgb, rel, 224)
    inputs = [view, prior, global_core, base_core, rgb_core]
    old_view = roi_sample(state.expand(len(ids), -1, -1, -1), views, 14)
    small = packet["small_rgb"][None].to(device).float() / 255
    small = (small - MEAN.to(device)) / STD.to(device)
    old_rgb = roi_sample(roi_sample(small.expand(len(ids), -1, -1, -1), views, 224), rel, 224)
    return ((*inputs, rel, old_view, old_rgb), boxes)


def corrected(packet, delta, boxes, device):
    height, width = packet["mask"].shape[-2:]
    base = F.interpolate(
        packet["logits"][None].float().to(device),
        (height, width),
        mode="bilinear",
        align_corners=False,
    )
    return base.expand(len(boxes), -1, -1, -1) + paste_roi(
        delta, boxes, (height, width), (1.0, 1.0)
    )


def initial_input(packet, device):
    state = packet["state"][None].float().to(device)
    p = packet["logits"][None].float().to(device).sigmoid().clamp(1e-06, 1 - 1e-06)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
    rgb = packet["small_rgb"][None].float().to(device) / 127.5 - 1
    return torch.cat(
        (
            state,
            F.adaptive_avg_pool2d(p, 22),
            F.adaptive_avg_pool2d(entropy, 22),
            F.adaptive_avg_pool2d(rgb, 22),
        ),
        1,
    )


def defect_target(packet, device):
    target = packet["mask"][None].bool().to(device)
    logits = F.interpolate(
        packet["logits"][None].float().to(device),
        target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    pred = logits >= 0
    classes = torch.cat((pred == target, pred & ~target, ~pred & target), 1).float()
    return F.adaptive_avg_pool2d(classes, 22)


def map_descriptor(maps, boxes):
    local = roi_sample(maps.expand(len(boxes), -1, -1, -1), boxes.to(maps.device), 14)
    grid = F.adaptive_avg_pool2d(local, 4).flatten(1)
    return torch.cat((grid, local.mean((2, 3)), local.std((2, 3))), 1)


def foreground_support(packet, device):
    if "native_size" in packet:
        size = packet["native_size"]
    else:
        from PIL import Image

        with Image.open(packet["source"]) as image:
            size = (image.height, image.width)
    logits = F.interpolate(
        packet["logits"][None].float().to(device), size, mode="bilinear", align_corners=False
    )
    return F.adaptive_avg_pool2d((logits >= 0).float(), 22)


def support_input(packet, device):
    return torch.cat((initial_input(packet, device), foreground_support(packet, device)), 1)


def conditional_loss(logits, target, support, weights):
    positive = target[:, 1:3].clamp_min(0)
    exposure = torch.cat((support, 1 - support), 1)
    negative = (exposure - positive).clamp_min(0)
    wp = weights[1:][None, :, None, None]
    wn = weights[0]
    return -(wp * positive * F.logsigmoid(logits) + wn * negative * F.logsigmoid(-logits)).sum() / (
        wp * positive + wn * negative
    ).sum().clamp_min(1e-08)


def features(packet, device, critic):
    maps = critic.probabilities(support_input(packet, device))
    return torch.cat((packet["descriptors"].to(device), map_descriptor(maps, packet["boxes"])), 1)


def loss_each(logits, target):
    weight = 1 + 5 * (F.avg_pool2d(target, 31, stride=1, padding=15) - target).abs()
    target = target.expand_as(logits)
    weight = weight.expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    bce = (weight * bce).sum((1, 2, 3)) / weight.sum((1, 2, 3))
    p = logits.sigmoid()
    intersection = (p * target * weight).sum((1, 2, 3))
    union = ((p + target) * weight).sum((1, 2, 3)) - intersection
    return bce + 1 - (intersection + 1) / (union + 1)


def dice_each(logits, target):
    p = logits >= 0
    g = target.bool().expand_as(p)
    return (2 * (p & g).sum((1, 2, 3)) + 1e-08) / (p.sum((1, 2, 3)) + g.sum((1, 2, 3)) + 1e-08)
