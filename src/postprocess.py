"""Mask refinement shared by the classical method and U-Net.

raw mask -> largest connected component -> hole filling -> ellipse fitting
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from src.data import ellipse_to_mask


def largest_component(mask):
    """Keeps only the largest 8-connected component."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def fill_holes(mask):
    return binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def fit_ellipse(mask):
    """Least-squares ellipse fitted on the outer contour: (cx, cy, a, b, angle_deg), or None.

    a, b are SEMI-axes, matching the FOCUS annotation format.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)  # w, h are FULL axes
    return (cx, cy, w / 2, h / 2, angle)


def refine(mask, to_ellipse=True):
    """Cleans a raw binary mask.

    Returns (refined_mask, ellipse). With to_ellipse=True the refined mask is the
    filled fitted ellipse; otherwise it is the cleaned component. ellipse is None
    when nothing usable was found.
    """
    clean = fill_holes(largest_component(mask))
    ellipse = fit_ellipse(clean)
    if ellipse is None:
        return clean, None
    return (ellipse_to_mask(mask.shape, ellipse) if to_ellipse else clean), ellipse
