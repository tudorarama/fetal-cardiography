"""Segmentation metrics: Dice and HD95 (95th percentile Hausdorff distance).

The same functions are used for the classical method and for U-Net, so both are
evaluated in exactly the same way.
"""
import warnings

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance

# MONAI 1.6 warns about an argument it passes to itself internally; not our code.
warnings.filterwarnings("ignore", message=".*always_return_as_numpy.*", category=FutureWarning)


def dice(pred, gt, eps=1e-7):
    """Dice similarity coefficient between two binary masks, in [0, 1]."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float(2 * (pred & gt).sum() / (pred.sum() + gt.sum() + eps))


def iou(pred, gt, eps=1e-7):
    """Intersection over union (Jaccard) between two binary masks, in [0, 1]."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((pred & gt).sum() / ((pred | gt).sum() + eps))


def hd95(pred, gt):
    """95th percentile Hausdorff distance between mask boundaries, in pixels.

    Returns nan when either mask is empty (the distance is undefined).
    """
    if not pred.any() or not gt.any():
        return float("nan")
    p = torch.from_numpy(pred.astype(np.float32)[None, None])
    g = torch.from_numpy(gt.astype(np.float32)[None, None])
    return float(compute_hausdorff_distance(p, g, include_background=True, percentile=95))
