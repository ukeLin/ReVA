"""Native-grid Dice/HD95, exact Euclidean nearest surface distances."""

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def metrics(prediction, target):
    p, g = prediction.astype(bool), target.astype(bool)
    intersection = np.count_nonzero(p & g)
    den = np.count_nonzero(p) + np.count_nonzero(g)
    union = np.count_nonzero(p | g)
    empty_p, empty_g = not p.any(), not g.any()
    diagonal = float(np.hypot(*p.shape))
    if empty_p or empty_g:
        hd95 = 0.0 if empty_p and empty_g else diagonal
    else:
        ps = p ^ binary_erosion(p, structure=np.ones((3, 3)), border_value=0)
        gs = g ^ binary_erosion(g, structure=np.ones((3, 3)), border_value=0)
        points_p = np.argwhere(ps)
        points_g = np.argwhere(gs)
        d1 = cKDTree(points_g).query(points_p, k=1, eps=0, p=2, workers=1)[0]
        d2 = cKDTree(points_p).query(points_g, k=1, eps=0, p=2, workers=1)[0]
        hd95 = float(np.percentile(np.concatenate((d1, d2)), 95))
    return {
        "dice": float((2 * intersection + 1e-8) / (den + 1e-8)),
        "iou": float((intersection + 1e-8) / (union + 1e-8)),
        "hd95_pixels": hd95,
        "hd95_diagonal_normalized": hd95 / diagonal,
        "empty_prediction": empty_p,
        "empty_target": empty_g,
        "false_positive_pixels": int(np.count_nonzero(p & ~g)),
        "false_negative_pixels": int(np.count_nonzero(~p & g)),
        "absolute_hard_area_error_fraction": float(
            abs(np.count_nonzero(p) - np.count_nonzero(g)) / p.size
        ),
    }
