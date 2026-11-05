import torch
from torch.nn import functional as F


def geometry(width, height):
    boxes, views = ([], [])
    for iy in range(3):
        for ix in range(3):
            x0, x1 = (round(ix * width / 3), round((ix + 1) * width / 3))
            y0, y1 = (round(iy * height / 3), round((iy + 1) * height / 3))
            boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
            for factor in (1.0, 1.5):
                vw, vh = (
                    min(width, round((x1 - x0) * factor)),
                    min(height, round((y1 - y0) * factor)),
                )
                vx = min(max(round((x0 + x1 - vw) / 2), 0), width - vw)
                vy = min(max(round((y0 + y1 - vh) / 2), 0), height - vh)
                views.append([vx / width, vy / height, (vx + vw) / width, (vy + vh) / height])
    return (torch.tensor(boxes), torch.tensor(views))


def relative(core, view):
    return torch.cat(
        (
            (core[:, :2] - view[:, :2]) / (view[:, 2:] - view[:, :2]),
            (core[:, 2:] - view[:, :2]) / (view[:, 2:] - view[:, :2]),
        ),
        1,
    )


def roi_sample(tensor, boxes, size=16, image_fraction=(1.0, 1.0)):
    centers = (torch.arange(size, device=tensor.device, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    x = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
    y = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
    grid = torch.stack([2 * x * image_fraction[1] - 1, 2 * y * image_fraction[0] - 1], dim=-1)
    return F.grid_sample(
        tensor.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def paste_roi(delta, boxes, shape, image_fraction):
    height, width = shape
    ys = (
        (torch.arange(height, device=delta.device, dtype=torch.float32) + 0.5)
        / height
        / image_fraction[0]
    )
    xs = (
        (torch.arange(width, device=delta.device, dtype=torch.float32) + 0.5)
        / width
        / image_fraction[1]
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (xx - boxes[:, 0, None, None]) / (boxes[:, 2] - boxes[:, 0])[:, None, None]
    y = (yy - boxes[:, 1, None, None]) / (boxes[:, 3] - boxes[:, 1])[:, None, None]
    grid = torch.stack([2 * x - 1, 2 * y - 1], dim=-1)
    support = ((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)).unsqueeze(1)
    return (
        F.grid_sample(
            delta.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        * support
    )
