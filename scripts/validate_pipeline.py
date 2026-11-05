"""Validates metrics, post-processing and measurements on the ground truth.

Run from the project root:
    python scripts/validate_pipeline.py

Before any segmentation method exists, the evaluation chain must give the
expected answers on ground-truth masks:
  1. metric sanity checks on synthetic masks;
  2. refine(GT mask) must reproduce the GT (Dice ~ 1) and the annotated CTR;
  3. refine() must repair a corrupted GT mask (noise blobs + holes).
Per-case results are saved to results/gt_measurements.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import ellipse_to_mask, list_cases, load_case
from src.measurements import cardiac_axis, ctr_area, ctr_circumference, ctr_from_masks
from src.metrics import dice, hd95
from src.postprocess import refine


def check_metrics():
    print("=== 1. Metric sanity checks")
    m = ellipse_to_mask((400, 400), (200, 200, 80, 50, 30))
    shifted = np.roll(m, 10, axis=1)
    checks = {
        "dice(m, m) == 1": abs(dice(m, m) - 1) < 1e-6,
        "hd95(m, m) == 0": hd95(m, m) == 0,
        "hd95(m, m shifted 10px) ~ 10": abs(hd95(m, shifted) - 10) < 1.5,
        "dice(m, empty) == 0": dice(m, np.zeros_like(m)) == 0,
        "hd95(m, empty) is nan": np.isnan(hd95(m, np.zeros_like(m))),
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"  hd95 after a 10 px shift: {hd95(m, shifted):.2f}")
    return all(checks.values())


def corrupt(mask, rng):
    """GT mask with random blobs outside and holes inside, like a noisy raw segmentation."""
    noisy = mask.copy()
    h, w = mask.shape
    for _ in range(8):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        blob = ellipse_to_mask(mask.shape, (cx, cy, rng.integers(5, 20), rng.integers(5, 20), 0))
        noisy[(blob > 0) & (mask == 0)] = 1
    ys, xs = np.nonzero(mask)
    for i in rng.choice(len(ys), 6, replace=False):
        hole = ellipse_to_mask(mask.shape, (xs[i], ys[i], 6, 6, 0))
        noisy[(hole > 0)] = 0
    return noisy


def check_on_ground_truth():
    print("\n=== 2-3. Post-processing and measurements on ground truth")
    rng = np.random.default_rng(0)
    rows = []
    for split in ("train", "val", "test"):
        for name in list_cases(split):
            img, ell, gt = load_case(split, name)
            row = {"split": split, "name": name}
            fitted = {}
            for k in ("cardiac", "thorax"):
                ref, fitted[k] = refine(gt[k])
                noisy = corrupt(gt[k], rng)
                ref_noisy, _ = refine(noisy)
                row[f"dice_refined_{k}"] = dice(ref, gt[k])
                row[f"dice_noisy_{k}"] = dice(noisy, gt[k])
                row[f"dice_noisy_refined_{k}"] = dice(ref_noisy, gt[k])
                row[f"hd95_noisy_{k}"] = hd95(noisy, gt[k])
                row[f"hd95_noisy_refined_{k}"] = hd95(ref_noisy, gt[k])
            row["ctr_area_gt"] = ctr_area(ell["cardiac"], ell["thorax"])
            row["ctr_area_fitted"] = ctr_area(fitted["cardiac"], fitted["thorax"])
            row["ctr_masks"] = ctr_from_masks(gt["cardiac"], gt["thorax"])
            row["ctr_circ_gt"] = ctr_circumference(ell["cardiac"], ell["thorax"])
            row["axis_gt"] = cardiac_axis(ell["cardiac"], ell["thorax"])
            row["axis_fitted"] = cardiac_axis(fitted["cardiac"], fitted["thorax"])
            row["thorax_axis_ratio"] = min(ell["thorax"][2:4]) / max(ell["thorax"][2:4])
            rows.append(row)

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "gt_measurements.csv", index=False)

    for k in ("cardiac", "thorax"):
        print(f"  {k}:")
        print(f"    refine(GT) vs GT          Dice {df[f'dice_refined_{k}'].mean():.4f}")
        print(f"    corrupted GT              Dice {df[f'dice_noisy_{k}'].mean():.4f}, "
              f"HD95 {df[f'hd95_noisy_{k}'].mean():6.1f} px")
        print(f"    refine(corrupted GT)      Dice {df[f'dice_noisy_refined_{k}'].mean():.4f}, "
              f"HD95 {df[f'hd95_noisy_refined_{k}'].mean():6.1f} px")
    d_ctr = (df["ctr_area_fitted"] - df["ctr_area_gt"]).abs()
    print(f"  CTR area: annotated {df['ctr_area_gt'].mean():.3f}, from fitted ellipses "
          f"{df['ctr_area_fitted'].mean():.3f} (max abs diff {d_ctr.max():.4f}), "
          f"from mask pixels {df['ctr_masks'].mean():.3f}")
    print(f"  CTR circumference (annotated): {df['ctr_circ_gt'].mean():.3f}")
    reliable = df["thorax_axis_ratio"] < 0.95
    d_axis = (df["axis_fitted"] - df["axis_gt"]).abs()
    print(f"  Cardiac axis: mean {df['axis_gt'].mean():.1f} deg; thorax clearly elliptical "
          f"(b/a < 0.95) in {reliable.sum()}/{len(df)} cases; "
          f"median |fitted - annotated| {d_axis.median():.2f} deg "
          f"({d_axis[reliable].median():.2f} deg on those cases)")
    print(f"\n  per-case table saved to {ROOT / 'results' / 'gt_measurements.csv'}")
    return df


if __name__ == "__main__":
    ok = check_metrics()
    check_on_ground_truth()
    if not ok:
        sys.exit("metric sanity checks failed")
