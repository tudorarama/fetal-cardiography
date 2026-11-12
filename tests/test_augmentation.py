"""Unit tests for training data augmentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.augmentation import augment


def test_augment_preserves_shape_dtype_and_binary_mask():
    rng = np.random.default_rng(42)
    img = rng.normal(0.0, 1.0, size=(1, 256, 256)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (40, 30), 0, 0, 360, 1, -1)

    cfg = {"p": 1.0, "max_angle": 15.0, "zoom_range": (0.9, 1.1), "brightness": 0.15}
    aug_img, aug_mask = augment(img, mask, cfg=cfg, rng=rng)

    assert aug_img.shape == (1, 256, 256)
    assert aug_img.dtype == np.float32
    assert aug_mask.shape == (256, 256)
    assert set(np.unique(aug_mask)).issubset({0, 1})
    assert not np.array_equal(aug_mask, mask)


def test_augment_supports_torch_tensors():
    import torch

    rng = np.random.default_rng(7)
    img = torch.randn((1, 256, 256), dtype=torch.float32)
    mask = torch.zeros((256, 256), dtype=torch.int64)
    mask[90:160, 100:170] = 1

    cfg = {"p": 1.0, "max_angle": 10.0, "zoom_range": (0.95, 1.05), "brightness": 0.1}
    aug_img, aug_mask = augment(img, mask, cfg=cfg, rng=rng)

    assert tuple(aug_img.shape) == (1, 256, 256)
    assert aug_img.dtype == torch.float32
    assert tuple(aug_mask.shape) == (256, 256)
    assert aug_mask.dtype == torch.int64
    assert set(aug_mask.unique().tolist()).issubset({0, 1})


def main():
    for fn in [test_augment_preserves_shape_dtype_and_binary_mask, test_augment_supports_torch_tensors]:
        fn()
        print(f"ok  {fn.__name__}")


if __name__ == "__main__":
    main()