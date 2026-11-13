"""Metrics and clinical measurements on four-chamber label maps.

Segmentation metrics (per chamber):
    Dice   2|P n G| / (|P| + |G|), overlap in [0, 1]
    HD95   95th percentile of the distances between the two contours (pixels):
           how far the worst part of the outline is, ignoring the 5 % extreme points

Measurements used for anomaly screening (rules, see screen()):
    LV/RV and LA/RA area ratios   ventricular / atrial disproportion
                                  (hypoplastic left or right heart, coarctation, ...)
    chamber fraction              a chamber < 3 % of the heart: missing or very small
                                  (single ventricle, AV septal defect merging chambers)
    cardiac axis                  angle between the interventricular septum and the
                                  spine -> sternum line, normal 45 +- 20 degrees; needs
                                  the optional spine/aorta landmark of the annotation
The thresholds are textbook starting points; they should be checked against the
range measured on the annotated (normal) images.
"""
import math

import cv2
import numpy as np
from scipy import ndimage

from config import CHAMBERS, LABELS

AXIS_RANGE = (25.0, 65.0)
RATIO_RANGE = (0.5, 2.0)
MIN_CHAMBER_FRACTION = 0.03


# --------------------------------------------------------------------- metrics

def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / s) if s else float("nan")


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred, gt):
    """95th percentile symmetric surface distance in pixels (nan if a mask is empty)."""
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndimage.distance_transform_edt(~sg)[sp]
    d_to_p = ndimage.distance_transform_edt(~sp)[sg]
    return float(np.percentile(np.concatenate([d_to_g, d_to_p]), 95))


# ---------------------------------------------------------------- measurements

def septum_direction(lab):
    """Unit direction (dx, dy) of the interventricular septum, or None.

    The septum is the tissue band between the LV and RV cavities, so the two
    annotated cavities do not touch: its midline is the set of pixels equidistant
    from both (within a quarter of the ventricles' size), and a line is fitted to it.
    """
    lv, rv = lab == LABELS["LV"], lab == LABELS["RV"]
    if lv.sum() == 0 or rv.sum() == 0:
        return None
    d_lv = cv2.distanceTransform((~lv).astype(np.uint8), cv2.DIST_L2, 5)
    d_rv = cv2.distanceTransform((~rv).astype(np.uint8), cv2.DIST_L2, 5)
    reach = 0.25 * np.sqrt(lv.sum() + rv.sum())
    ys, xs = np.nonzero((np.abs(d_lv - d_rv) <= 1.5) & (np.maximum(d_lv, d_rv) < reach))
    if len(xs) >= 10:
        vx, vy, _, _ = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32),
                                   cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return float(vx), float(vy)
    (ly, lx), (ry, rx) = [np.argwhere(m).mean(axis=0) for m in (lv, rv)]
    return float(-(ry - ly)), float(rx - lx)  # perpendicular to the LV -> RV vector


def cardiac_axis(lab, landmark_xy, thorax_center):
    """Angle (deg, 0-90) between the septum and the antero-posterior line.

    landmark_xy: spine (or descending aorta) centre; thorax_center: centre of the
    thorax (FOCUS thorax mask). The line between them is the spine -> sternum
    direction. (The heart centre cannot replace the thorax centre: the heart lies
    left of the midline, which would bias the angle.)
    """
    sept = septum_direction(lab)
    if sept is None or landmark_xy is None or thorax_center is None:
        return None
    cx, cy = thorax_center
    ap = (cx - landmark_xy[0], cy - landmark_xy[1])
    a = abs(math.degrees(math.atan2(sept[1], sept[0])) - math.degrees(math.atan2(ap[1], ap[0])))
    a %= 180
    return float(min(a, 180 - a))


def chamber_measurements(lab, landmark_xy=None, thorax_center=None):
    area = {c: float((lab == LABELS[c]).sum()) for c in CHAMBERS}
    total = sum(area.values())
    return {"areas": area,
            "fractions": {c: (a / total if total else 0.0) for c, a in area.items()},
            "lv_rv": area["LV"] / area["RV"] if area["RV"] else None,
            "la_ra": area["LA"] / area["RA"] if area["RA"] else None,
            "axis": cardiac_axis(lab, landmark_xy, thorax_center)}


def screen(m):
    """Anomaly flags (list of codes) from chamber_measurements(); empty list = no flag."""
    flags = []
    for c, f in m["fractions"].items():
        if f < MIN_CHAMBER_FRACTION:
            flags.append(f"missing_{c}")
    for key, code in (("lv_rv", "ventricle_disproportion"), ("la_ra", "atrium_disproportion")):
        v = m[key]
        if v is not None and not RATIO_RANGE[0] <= v <= RATIO_RANGE[1]:
            flags.append(code)
    if m["axis"] is not None and not AXIS_RANGE[0] <= m["axis"] <= AXIS_RANGE[1]:
        flags.append("abnormal_axis")
    return flags
