"""Ultrasound image preprocessing, ROI cropping, and U-Net tensor formatting."""

import cv2
import numpy as np

from src.config import NORM_MEAN, NORM_STD, PREPROC, UNET_SIZE


def preprocess(img, cfg=None):
    """Apply speckle denoising and local contrast enhancement to a grayscale image."""
    cfg = PREPROC if cfg is None else cfg
    out = img.copy()

    k = cfg.get("median", 5)
    if k and k > 1:
        out = cv2.medianBlur(out, k)

    if cfg.get("bilateral", True):
        out = cv2.bilateralFilter(
            out,
            d=cfg.get("bilateral_d", 9),
            sigmaColor=cfg.get("bilateral_sigma_color", 50),
            sigmaSpace=cfg.get("bilateral_sigma_space", 50),
        )

    clip = cfg.get("clahe", 2.0)
    if clip and clip > 0:
        tile = cfg.get("clahe_tile", 8)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
        out = clahe.apply(out)

    return out


def _letterbox(img, size, interp=cv2.INTER_LINEAR, pad_val=0):
    """Resize while preserving aspect ratio and pad symmetrically to (size, size)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h = round(h * scale)
    new_w = round(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_val
    )
    meta = {
        "orig_shape": (h, w),
        "scale": scale,
        "pad": (top, bottom, left, right),
    }
    return padded, meta


def to_unet_input(img, size=UNET_SIZE):
    """Letterbox a grayscale image to (size, size) and scale to [0, 1] float32."""
    padded, meta = _letterbox(img, size, interp=cv2.INTER_LINEAR, pad_val=0)
    return (padded.astype(np.float32) / 255.0), meta


def mask_to_unet(mask, size=UNET_SIZE):
    """Letterbox a binary mask to (size, size) using nearest-neighbor interpolation."""
    padded, _ = _letterbox(mask.astype(np.uint8), size, interp=cv2.INTER_NEAREST, pad_val=0)
    return (padded > 0).astype(np.uint8)


def restore_size(mask, meta):
    """Remove letterbox padding and resize a mask back to its original shape."""
    top, bottom, left, right = meta["pad"]
    orig_h, orig_w = meta["orig_shape"]
    h, w = mask.shape[:2]
    unpadded = mask[top : h - bottom, left : w - right]
    return cv2.resize(
        unpadded.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )


def crop_heart_roi(img, heart_mask, margin=0.25):
    """Crop a padded bounding box around the cardiac mask and return (crop, roi_meta)."""
    h, w = img.shape[:2]
    ys, xs = np.where(heart_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        roi_meta = {"orig_shape": (h, w), "box": (0, 0, w, h)}
        return img.copy(), roi_meta

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    pad_x = round((x_max - x_min) * margin)
    pad_y = round((y_max - y_min) * margin)

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x)
    y1 = min(h, y_max + pad_y)

    roi_meta = {"orig_shape": (h, w), "box": (x0, y0, x1, y1)}
    return img[y0:y1, x0:x1].copy(), roi_meta


def paste_back(cropped_mask, roi_meta):
    """Place a cropped mask back into the full original image frame."""
    full = np.zeros(roi_meta["orig_shape"], dtype=cropped_mask.dtype)
    x0, y0, x1, y1 = roi_meta["box"]
    full[y0:y1, x0:x1] = cropped_mask
    return full


def prepare(img, heart_mask=None, cfg=None, size=UNET_SIZE, return_meta=False):
    """Full preprocessing pipeline returning a (1, size, size) float32 tensor."""
    import torch

    cfg = PREPROC if cfg is None else cfg
    h, w = img.shape[:2]

    if cfg.get("crop", False) and heart_mask is not None:
        out, roi_meta = crop_heart_roi(img, heart_mask, margin=cfg.get("roi_margin", 0.25))
    else:
        out = img
        roi_meta = {"orig_shape": (h, w), "box": (0, 0, w, h)}

    if cfg.get("filter", True):
        out = preprocess(out, cfg)

    x, resize_meta = to_unet_input(out, size=size)

    if cfg.get("normalize", "01") == "zscore":
        mean = cfg.get("mean", NORM_MEAN) / 255.0
        std = cfg.get("std", NORM_STD) / 255.0
        x = ((x - mean) / std).astype(np.float32)

    tensor = torch.from_numpy(x).unsqueeze(0)
    if return_meta:
        return tensor, {"roi": roi_meta, "resize": resize_meta}
    return tensor