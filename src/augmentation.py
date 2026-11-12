"""On-the-fly training augmentations synchronized across image and mask."""

import cv2
import numpy as np

from src.config import AUGMENT


def augment(image, mask, cfg=None, rng=None):
    """Apply random small rotation, zoom, and brightness shift to a training pair."""
    cfg = AUGMENT if cfg is None else cfg
    rng = np.random.default_rng() if rng is None else rng

    if rng.random() > cfg.get("p", 0.8):
        return image, mask

    is_tensor_img = hasattr(image, "detach")
    is_tensor_mask = hasattr(mask, "detach")

    img_np = image.detach().cpu().numpy() if is_tensor_img else np.asarray(image)
    mask_np = mask.detach().cpu().numpy() if is_tensor_mask else np.asarray(mask)

    img_has_ch = img_np.ndim == 3
    mask_has_ch = mask_np.ndim == 3

    img_2d = img_np[0].astype(np.float32) if img_has_ch else img_np.astype(np.float32)
    # every channel of the mask is kept (e.g. heart + thorax), each warped the same way
    mask_chs = mask_np.astype(np.uint8) if mask_has_ch else mask_np.astype(np.uint8)[np.newaxis]

    h, w = img_2d.shape
    max_angle = float(cfg.get("max_angle", 15.0))
    z_min, z_max = cfg.get("zoom_range", (0.9, 1.1))

    angle = float(rng.uniform(-max_angle, max_angle))
    scale = float(rng.uniform(z_min, z_max))

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    bg_val = float(img_2d.min())

    aug_img = cv2.warpAffine(
        img_2d,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_val,
    )
    # nearest neighbour keeps the label values (0/1 masks, or classes 0-4 for the chambers)
    aug_mask = np.stack([
        cv2.warpAffine(
            ch,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for ch in mask_chs
    ])

    bright = float(cfg.get("brightness", 0.15))
    if bright > 0:
        delta = float(rng.uniform(-bright, bright)) * max(float(aug_img.std()), 1e-3)
        aug_img = (aug_img + delta).astype(np.float32)

    aug_mask = aug_mask.astype(mask_np.dtype)
    if not mask_has_ch:
        aug_mask = aug_mask[0]

    if img_has_ch:
        aug_img = aug_img[np.newaxis, ...]

    if is_tensor_img:
        import torch
        aug_img = torch.from_numpy(aug_img).to(device=image.device, dtype=image.dtype)
    if is_tensor_mask:
        import torch
        aug_mask = torch.from_numpy(aug_mask).to(device=mask.device, dtype=mask.dtype)

    return aug_img, aug_mask