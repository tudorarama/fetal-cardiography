"""Unit tests for post-processing, metrics and measurements."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import ellipse_to_mask
from src.measurements import cardiac_axis, ctr_area
from src.metrics import dice, hd95
from src.postprocess import refine

SHAPE = (300, 400)
HEART = (200.0, 150.0, 40.0, 30.0, 20.0)
THORAX = (200.0, 150.0, 110.0, 90.0, 0.0)


def test_dice_and_hd95_of_identical_masks():
    m = ellipse_to_mask(SHAPE, HEART)
    assert dice(m, m) > 0.999
    assert hd95(m, m) == 0.0
    assert np.isnan(hd95(m, np.zeros_like(m)))


def test_refine_removes_noise_and_holes():
    m = ellipse_to_mask(SHAPE, HEART)
    noisy = m.copy()
    noisy[10:20, 10:20] = 1          # separate blob
    noisy[148:152, 198:202] = 0      # hole inside
    clean, ellipse = refine(noisy)
    assert ellipse is not None
    assert dice(clean, m) > 0.98
    assert clean[10:20, 10:20].sum() == 0


def test_ctr_and_axis_from_ellipses():
    assert abs(ctr_area(HEART, THORAX) - (40 * 30) / (110 * 90)) < 1e-9
    assert abs(cardiac_axis(HEART, THORAX) - 20.0) < 1e-9


def main():
    for fn in [test_dice_and_hd95_of_identical_masks, test_refine_removes_noise_and_holes,
               test_ctr_and_axis_from_ellipses]:
        fn()
        print(f"ok  {fn.__name__}")


if __name__ == "__main__":
    main()
