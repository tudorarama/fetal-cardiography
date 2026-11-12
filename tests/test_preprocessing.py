"""Unit tests for the preprocessing pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.preprocessing import (
    crop_heart_roi,
    mask_to_unet,
    paste_back,
    prepare,
    preprocess,
    restore_size,
    to_unet_input,
)


def _fake_image(h=392, w=640):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_preprocess_preserves_shape_and_dtype():
    img = _fake_image()
    out = preprocess(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_to_unet_input_shape_and_range():
    img = _fake_image()
    x, meta = to_unet_input(img, size=256)
    assert x.shape == (256, 256)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert meta["orig_shape"] == img.shape


def test_mask_to_unet_is_binary():
    mask = np.zeros((392, 640), dtype=np.uint8)
    mask[100:200, 150:300] = 255
    out = mask_to_unet(mask, size=256)
    assert out.shape == (256, 256)
    assert set(np.unique(out)).issubset({0, 1})


def test_restore_size_roundtrip():
    img = _fake_image(392, 640)
    _, meta = to_unet_input(img, size=256)

    mask = np.zeros((392, 640), dtype=np.uint8)
    cv2.ellipse(mask, (320, 196), (90, 70), 0, 0, 360, 1, -1)

    small = mask_to_unet(mask, size=256)
    restored = restore_size(small, meta)
    assert restored.shape == (392, 640)

    inter = np.logical_and(mask > 0, restored > 0).sum()
    dice = 2.0 * inter / ((mask > 0).sum() + (restored > 0).sum())
    assert dice > 0.98


def test_crop_and_paste_back_roundtrip():
    h, w = 300, 420
    img = _fake_image(h, w)
    heart_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(heart_mask, (210, 150), (50, 40), 0, 0, 360, 1, -1)

    crop, roi_meta = crop_heart_roi(img, heart_mask, margin=0.25)
    assert crop.shape[0] < h and crop.shape[1] < w

    x0, y0, x1, y1 = roi_meta["box"]
    restored = paste_back(heart_mask[y0:y1, x0:x1], roi_meta)
    assert restored.shape == (h, w)
    assert np.array_equal(restored, heart_mask)


def test_prepare_returns_unet_tensor():
    img = _fake_image()
    t = prepare(img)
    assert tuple(t.shape) == (1, 256, 256)
    assert str(t.dtype) == "torch.float32"


def main():
    tests = [
        test_preprocess_preserves_shape_and_dtype,
        test_to_unet_input_shape_and_range,
        test_mask_to_unet_is_binary,
        test_restore_size_roundtrip,
        test_crop_and_paste_back_roundtrip,
        test_prepare_returns_unet_tensor,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")


if __name__ == "__main__":
    main()