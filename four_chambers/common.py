"""Shared helpers: FOCUS images, chamber annotations, label maps, crops."""
import json
from pathlib import Path

import cv2
import numpy as np

from config import ANN_DIR, CHAMBERS, FOCUS_DIR, LABELS, LANDMARKS, SPLITS


# ------------------------------------------------------------------- FOCUS data

def list_cases(split):
    """Case names ('001', ...) of an official FOCUS split: 'train', 'val' or 'test'."""
    return sorted(p.stem for p in (FOCUS_DIR / SPLITS[split] / "images").glob("*.png"))


def image_path(split, name):
    return FOCUS_DIR / SPLITS[split] / "images" / f"{name}.png"


def load_image(split, name):
    """Grayscale uint8 (H, W)."""
    img = cv2.imread(str(image_path(split, name)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path(split, name))
    return img


def load_focus_masks(split, name):
    """FOCUS heart ('cardiac') and thorax masks, 0/1 uint8. Used only to frame the view."""
    d = FOCUS_DIR / SPLITS[split] / "annfiles_mask"
    return {k: (cv2.imread(str(d / f"{name}-{k}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            for k in ("cardiac", "thorax")}


def clahe(img, clip=3.0, grid=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def case_id(split, name):
    """nnU-Net case identifier, e.g. 'train_106'."""
    return f"{split}_{name}"


# ------------------------------------------------------------------ annotations

def annotation_path(split, name, ann_dir=ANN_DIR):
    return Path(ann_dir) / f"{split}_{name}.json"


def load_annotation(split, name, ann_dir=ANN_DIR):
    p = annotation_path(split, name, ann_dir)
    return json.loads(p.read_text()) if p.exists() else None


def save_annotation(ann, ann_dir=ANN_DIR):
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    annotation_path(ann["split"], ann["name"], ann_dir).write_text(json.dumps(ann, indent=1))


def is_complete(ann):
    """All 4 chambers drawn and the image not marked as unusable (landmarks optional)."""
    return ann is not None and not ann.get("skipped") and \
        all(len(ann["polygons"].get(c, [])) >= 3 for c in CHAMBERS)


def completed(ann_dir=ANN_DIR):
    """[(split, name)] of every completely annotated image."""
    out = []
    for p in sorted(Path(ann_dir).glob("*_*.json")):
        if p.stem == "selection":
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
    """Square box (x0, y0, side) around the FOCUS heart outline (the region of interest).

    The chambers are inside the heart, so cropping there removes the text overlay,
    the sector border and most of the image, and makes training ~5x cheaper.
    FOCUS provides the heart outline for every image (train, val and test).
    """
    ys, xs = np.nonzero(load_focus_masks(split, name)["cardiac"])
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
