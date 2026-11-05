"""Sanity check of the downloaded datasets.

Run from the project root:
    python scripts/check_data.py

Checks that every FOCUS case loads, that the ellipse annotations match the
ground-truth masks, prints dataset statistics and saves an overlay figure to
results/focus_overlay.png.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.data import (STRUCTURES, ellipse_to_mask, list_cases, load_case,
                      load_planes_table, overlay)


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def check_focus():
    print("=== FOCUS")
    sizes = Counter()
    agreement = {k: [] for k in STRUCTURES}
    ctr = []
    for split in ("train", "val", "test"):
        names = list_cases(split)
        print(f"{split:5s}: {len(names)} cases")
        for name in names:
            img, ell, masks = load_case(split, name)
            sizes[img.shape] += 1
            for k in STRUCTURES:
                assert masks[k].shape == img.shape, f"{split}/{name} {k}: mask/image size mismatch"
                agreement[k].append(dice(ellipse_to_mask(img.shape, ell[k]), masks[k]))
            h, t = ell["cardiac"], ell["thorax"]
            ctr.append(h[2] * h[3] / (t[2] * t[3]))

    print("image sizes (H, W):", dict(sizes.most_common()))
    for k, v in agreement.items():
        print(f"ellipse vs mask Dice, {k:7s}: mean {np.mean(v):.4f}, min {np.min(v):.4f}")
    ctr = np.array(ctr)
    print(f"CTR (area ratio): mean {ctr.mean():.3f} +- {ctr.std():.3f}, "
          f"range [{ctr.min():.3f}, {ctr.max():.3f}], > 0.35: {(ctr > 0.35).sum()} cases")


def check_planes():
    print("\n=== FETAL_PLANES_DB")
    df = load_planes_table()
    missing = sum(not p.exists() for p in df["image_path"])
    print(f"{len(df)} images, {missing} missing files")
    print(df.groupby(["Plane", "Train"]).size().unstack().rename(columns={0: "test", 1: "train"}))


def save_overlay(path=ROOT / "results" / "focus_overlay.png", names=("001", "002", "003")):
    tiles = []
    for name in names:
        img, _, masks = load_case("train", name)
        vis = overlay(img, masks)
        tiles.append(cv2.resize(vis, (480, round(480 * vis.shape[0] / vis.shape[1]))))
    h = min(t.shape[0] for t in tiles)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([t[:h] for t in tiles]), cv2.COLOR_RGB2BGR))
    print(f"\noverlay saved to {path}")


if __name__ == "__main__":
    check_focus()
    check_planes()
    save_overlay()
