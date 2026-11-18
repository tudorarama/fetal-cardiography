"""Compute the z-score statistics (mean, std) of the model input, on the TRAIN split only.

The statistics must describe exactly what the network receives: the image after the
steps enabled in PREPROC (ROI crop, filters) and after the letterbox to UNET_SIZE.
Computing them on raw images and applying them after the filters gives inputs that
are not centred (mean != 0, std != 1).

Run from the project root:
    python scripts/compute_norm_stats.py               # uses PREPROC from src/config.py
    python scripts/compute_norm_stats.py --no-filter   # statistics for the "raw" ablation
    python scripts/compute_norm_stats.py --crop        # statistics with the heart ROI crop

Then paste the printed NORM_MEAN / NORM_STD into src/config.py and regenerate
data/processed/ (python scripts/preprocess_dataset.py). Whenever PREPROC changes,
run this script again.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import PREPROC, UNET_SIZE
from src.data import list_cases, load_case
from src.preprocessing import crop_heart_roi, preprocess, to_unet_input


def model_input(img, heart_mask, cfg):
    """Same steps as prepare(), without the normalization. Returns float values in 0-255."""
    out = img
    if cfg.get("crop", False) and heart_mask is not None:
        out, _ = crop_heart_roi(out, heart_mask, margin=cfg.get("roi_margin", 0.25))
    if cfg.get("filter", True):
        out = preprocess(out, cfg)
    x, _ = to_unet_input(out, size=UNET_SIZE)   # float32 in [0, 1], letterboxed
    return x.astype(np.float64) * 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-filter", action="store_true", help="statistics without the filters")
    ap.add_argument("--crop", action="store_true", help="statistics with the heart ROI crop")
    args = ap.parse_args()

    cfg = dict(PREPROC)
    if args.no_filter:
        cfg["filter"] = False
    if args.crop:
        cfg["crop"] = True

    total = s = s2 = 0.0
    for name in list_cases("train"):
        img, _, masks = load_case("train", name)
        x = model_input(img, masks["cardiac"], cfg)
        total += x.size
        s += x.sum()
        s2 += (x ** 2).sum()
    mean = s / total
    std = float(np.sqrt(s2 / total - mean ** 2))

    print(f"settings: filter={cfg.get('filter', True)}, crop={cfg.get('crop', False)}, "
          f"size={UNET_SIZE}, train images={len(list_cases('train'))}")
    print("paste into src/config.py:")
    print(f"NORM_MEAN = {mean:.2f}")
    print(f"NORM_STD = {std:.2f}")

    # check: normalizing the train images with these values must give mean 0, std 1
    t = t2 = n = 0.0
    for name in list_cases("train"):
        img, _, masks = load_case("train", name)
        z = (model_input(img, masks["cardiac"], cfg) - mean) / std
        t += z.sum(); t2 += (z ** 2).sum(); n += z.size
    print(f"check on train after normalization: mean={t / n:.3f}, std={np.sqrt(t2 / n - (t / n) ** 2):.3f}")


if __name__ == "__main__":
    main()