"""Shared helpers: images of the three sources, chamber annotations, label maps, crops.

A case is (split, name). FOCUS: split train / val / test, name '001'.
FETAL_PLANES_DB: split ptrain / pval / ptest, name 'Patient00168_Plane6_1_of_4'.
HEARTBEAT: split htrain / hval / htest, name '2T_P0123.4C.01' (example) (trimester_image).
role(split) gives 'train', 'val' or 'test' for every source.
"""
import functools
import json
from pathlib import Path

import cv2
import numpy as np

from config import (ANN_DIR, CHAMBERS, FOCUS_DIR, HEARTBEAT_ANN_DIR, HEARTBEAT_DIR, HEARTBEAT_MASK_DIR,
                    HEARTBEAT_SPLIT_FILE, HEARTBEAT_SPLITS, LABELS, LANDMARKS, PLANES_DIR,
                    PLANES_SPLIT_FILE, PLANES_SPLITS, SPLITS)


# ----------------------------------------------------------------- image data

def role(split):
    """'train', 'val' or 'test' for a split of any source."""
    return PLANES_SPLITS.get(split) or HEARTBEAT_SPLITS.get(split) or split


def is_planes(split):
    return split in PLANES_SPLITS


def is_heartbeat(split):
    return split in HEARTBEAT_SPLITS


def source(split):
    if is_heartbeat(split):
        return "HEARTBEAT"
    return "FETAL_PLANES_DB" if is_planes(split) else "FOCUS"


@functools.lru_cache(maxsize=4)
def _heartbeat_split(mtime):
    return json.loads(HEARTBEAT_SPLIT_FILE.read_text())


def heartbeat_split():
    """select_heartbeat.py output: {'htrain': [...], 'hval': [...], 'htest': [...], 'paths': {name: rel}}."""
    if not HEARTBEAT_SPLIT_FILE.exists():
        return {}
    return _heartbeat_split(HEARTBEAT_SPLIT_FILE.stat().st_mtime)


def heartbeat_rel_path(row):
    """'<trimester>/<part>/<class>/<patient>/<image>.png' of a row of heartbeat_outline.csv."""
    stem = row["id"].split("_", 1)[1]
    cls = "CHD" if row["chd"] else "No_CHD"
    return f"{row['trimester']}/{row['part']}/{cls}/{row['patient']}/{stem}.png"


def list_cases(split):
    """Case names of a split: FOCUS folders, or the chosen FETAL_PLANES_DB / Heartbeat images."""
    if is_heartbeat(split):
        return list(heartbeat_split().get(split, []))
    if is_planes(split):
        if not PLANES_SPLIT_FILE.exists():
            return []
        return list(json.loads(PLANES_SPLIT_FILE.read_text())[split])
    return sorted(p.stem for p in (FOCUS_DIR / SPLITS[split] / "images").glob("*.png"))


def image_path(split, name):
    if is_heartbeat(split):
        return HEARTBEAT_DIR / heartbeat_split()["paths"][name]
    if is_planes(split):
        return PLANES_DIR / "Images" / f"{name}.png"
    return FOCUS_DIR / SPLITS[split] / "images" / f"{name}.png"


def short_name(split, name):
    """Short label for small displays: 'train 106', 'ptest P168 6_1/4'."""
    if is_planes(split) and name.startswith("Patient"):
        pat, _, rest = name.partition("_Plane")
        return f"{split} P{int(pat[7:])} {rest.replace('_of_', '/')}"
    return f"{split} {name}"


def load_image(split, name):
    """Grayscale uint8 (H, W)."""
    img = cv2.imread(str(image_path(split, name)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path(split, name))
    return img


def load_focus_masks(split, name):
    """Heart ('cardiac') and thorax masks, 0/1 uint8: the FOCUS ground truth, or for
    FETAL_PLANES_DB the stage-1 U-Net prediction. Used to frame the view and the ROI."""
    if is_heartbeat(split):
        d = HEARTBEAT_MASK_DIR
    else:
        d = PLANES_DIR / "masks" if is_planes(split) else FOCUS_DIR / SPLITS[split] / "annfiles_mask"
    return {k: (cv2.imread(str(d / f"{name}-{k}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            for k in ("cardiac", "thorax")}


def clahe(img, clip=3.0, grid=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def case_id(split, name):
    """nnU-Net case identifier, e.g. 'train_106'."""
    return f"{split}_{name}"


# ------------------------------------------------------------------ annotations

def annotation_dir(split):
    """Heartbeat annotations are kept apart (not versioned: restricted data)."""
    return HEARTBEAT_ANN_DIR if is_heartbeat(split) else ANN_DIR


def annotation_path(split, name, ann_dir=None):
    return Path(ann_dir or annotation_dir(split)) / f"{split}_{name}.json"


def load_annotation(split, name, ann_dir=None):
    p = annotation_path(split, name, ann_dir)
    return json.loads(p.read_text()) if p.exists() else None


def save_annotation(ann, ann_dir=None):
    p = annotation_path(ann["split"], ann["name"], ann_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ann, indent=1))


def is_complete(ann):
    """All 4 chambers drawn and the image not marked as unusable (landmarks optional)."""
    return ann is not None and not ann.get("skipped") and \
        all(len(ann["polygons"].get(c, [])) >= 3 for c in CHAMBERS)


def completed(ann_dir=None):
    """[(split, name)] of every completely annotated image (default: all sources)."""
    out = []
    dirs = [Path(ann_dir)] if ann_dir else [ANN_DIR, HEARTBEAT_ANN_DIR]
    for p in sorted(f for d in dirs for f in d.glob("*_*.json")):
        if p.stem in ("selection", "heartbeat_split"):
            continue
        ann = json.loads(p.read_text())
        if is_complete(ann):
            out.append((ann["split"], ann["name"]))
    return out


def rasterize(ann, shape):
    """Chamber polygons -> label map (H, W) uint8 with the values of config.LABELS."""
    lab = np.zeros(shape[:2], dtype=np.uint8)
    for c in CHAMBERS[::-1]:
        pts = ann["polygons"].get(c, [])
        if len(pts) >= 3:
            cv2.fillPoly(lab, [np.round(np.array(pts)).astype(np.int32)], LABELS[c])
    return lab


def landmark(ann):
    """(x, y, name) of the posterior landmark (spine, else aorta) or None."""
    polygons = (ann or {}).get("polygons", {})
    for name in LANDMARKS:
        pts = polygons.get(name, [])
        if len(pts) >= 3:
            x, y = np.mean(pts, axis=0)
            return float(x), float(y), name
    return None


def labels_to_polygons(lab, min_area=30):
    """Label map -> {chamber: polygon} (largest outer contour of each chamber)."""
    polys = {}
    for c in CHAMBERS:
        m = (lab == LABELS[c]).astype(np.uint8)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnt:
            big = max(cnt, key=cv2.contourArea)
            if cv2.contourArea(big) >= min_area:
                poly = cv2.approxPolyDP(big, 1.0, True)[:, 0]
                polys[c] = [[float(x), float(y)] for x, y in poly]
    return polys


# --------------------------------------------------------------------- geometry

ROI_SIZE = 384      # the network sees a ROI_SIZE x ROI_SIZE square around the heart
ROI_MARGIN = 0.2    # fraction of the heart size added on each side


def heart_roi_box(split, name, margin=ROI_MARGIN):
    """Square box (x0, y0, side) around the heart outline (the region of interest).

    The chambers are inside the heart, so cropping there removes the text overlay,
    the sector border and most of the image, and makes training ~5x cheaper.
    FOCUS provides the heart outline for every image (train, val and test); for
    FETAL_PLANES_DB it comes from the stage-1 U-Net, in training and at prediction alike.
    """
    return roi_box(load_focus_masks(split, name)["cardiac"], margin)


def roi_box(heart_mask, margin=ROI_MARGIN):
    """Square box (x0, y0, side) around a heart mask, enlarged by margin on each side."""
    ys, xs = np.nonzero(heart_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def uncrop(arr, box, shape, interpolation=cv2.INTER_LINEAR):
    """Inverse of crop: (size, size[, C]) array -> original (H, W[, C]), zero outside the box."""
    x0, y0, side = box
    big = cv2.resize(arr, (side, side), interpolation=interpolation)
    if big.ndim < arr.ndim:  # cv2 drops a single channel
        big = big[..., None]
    out = np.zeros(shape[:2] + arr.shape[2:], dtype=arr.dtype)
    h, w = shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    out[sy0:sy1, sx0:sx1] = big[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    return out


def thorax_crop_box(thorax_mask, margin=0.12):
    """Square box (x0, y0, side) around the thorax mask, may extend outside the image."""
    ys, xs = np.nonzero(thorax_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop(arr, box, size, interpolation=cv2.INTER_AREA):
    """Crops the square box (zero padding outside the image) and resizes to size x size."""
    x0, y0, side = box
    out = np.zeros((side, side) + arr.shape[2:], dtype=arr.dtype)
    h, w = arr.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (size, size), interpolation=interpolation)
