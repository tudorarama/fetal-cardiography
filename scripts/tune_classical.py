"""Chooses the parameters of the classical method.

Run from the project root (takes ~10-15 minutes on CPU):
    python scripts/tune_classical.py

  1. thorax parameters: small grid search on the VALIDATION set;
  2. heart (Chan-Vese) parameters: grid search on VALIDATION, with the thorax
     parameters from step 1;
  3. heart_scale (the heart ellipse is fitted on the chambers only, the GT also
     contains the myocardium): the value that maximises the mean Dice on TRAIN.

The test set is never used here. Output: results/classical_params.json and
results/classical_tuning.csv.
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classical import (WORK_SIZE, ClassicalParams, field_of_view, save_params,
                           segment, segment_thorax)
from src.data import ellipse_to_mask, list_cases, load_case
from src.metrics import dice
from src.preprocess import preprocess

OUT = ROOT / "results"

THORAX_GRID = {"wall_pick": [0.5, 0.6, 0.7, 0.8], "radial_sigma": [3.0, 5.0],
               "prior_spread": [0.25, 0.35]}
HEART_GRID = {"cv_levels": [2, 3], "heart_cluster": [0.5, 0.8],
              "heart_center_spread": [0.3, 0.5]}
SCALES = np.round(np.arange(0.8, 1.61, 0.05), 2)


def load_split(split):
    cases = []
    for name in tqdm(list_cases(split), desc=f"loading {split}"):
        img, ell, gt = load_case(split, name)
        cases.append((name, preprocess(img), gt))
    return cases


def grid(spec):
    keys = list(spec)
    for values in itertools.product(*(spec[k] for k in keys)):
        yield dict(zip(keys, values))


def thorax_dice(cases, p):
    scores = []
    for _, pre, gt in cases:
        h0, w0 = pre.shape
        s = WORK_SIZE / max(h0, w0)
        work = cv2.resize(pre, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        _, e = segment_thorax(work, field_of_view(work, p), p)
        if e is None:
            scores.append(0.0)
            continue
        e = (e[0] / s, e[1] / s, e[2] / s, e[3] / s, e[4])
        scores.append(dice(ellipse_to_mask(pre.shape, e), gt["thorax"]))
    return float(np.mean(scores))


def heart_runs(cases, p):
    """Segments every case once; heart Dice for any scale is then cheap to compute."""
    runs = []
    for _, pre, gt in cases:
        r = segment(pre, replace(p, heart_scale=1.0))
        runs.append((r, gt))
    return runs


def heart_dice_at(runs, scale):
    scores = []
    for r, gt in runs:
        if r.heart_unscaled is None:
            scores.append(0.0)
            continue
        cx, cy, a, b, ang = r.heart_unscaled
        m = ellipse_to_mask(gt["cardiac"].shape, (cx, cy, a * scale, b * scale, ang))
        scores.append(dice(m & r.masks["thorax"], gt["cardiac"]))
    return float(np.mean(scores))


def main():
    OUT.mkdir(exist_ok=True)
    log = []
    val = load_split("val")

    print("\n[1/3] thorax parameters on validation")
    best_t, best_p = -1, None
    for g in tqdm(list(grid(THORAX_GRID))):
        p = replace(ClassicalParams(), **g)
        d = thorax_dice(val, p)
        log.append({"stage": "thorax", **g, "val_dice_thorax": d})
        if d > best_t:
            best_t, best_p = d, p
    print(f"  best thorax Dice on val: {best_t:.3f}  "
          f"{ {k: getattr(best_p, k) for k in THORAX_GRID} }")

    print("\n[2/3] heart parameters on validation")
    best_h, best_hp = -1, None
    for g in tqdm(list(grid(HEART_GRID))):
        p = replace(best_p, **g)
        runs = heart_runs(val, p)
        d, sc = max((heart_dice_at(runs, s), s) for s in SCALES)
        log.append({"stage": "heart", **g, "val_dice_cardiac": d, "best_scale_val": sc})
        if d > best_h:
            best_h, best_hp = d, p
    print(f"  best heart Dice on val: {best_h:.3f}  "
          f"{ {k: getattr(best_hp, k) for k in HEART_GRID} }")

    print("\n[3/3] heart_scale on train")
    runs = heart_runs(load_split("train"), best_hp)
    curve = [(s, heart_dice_at(runs, s)) for s in SCALES]
    scale, d_train = max(curve, key=lambda t: t[1])
    for s, d in curve:
        log.append({"stage": "scale", "heart_scale": s, "train_dice_cardiac": d})
    final = replace(best_hp, heart_scale=float(scale))
    print(f"  heart_scale = {scale} (train heart Dice {d_train:.3f})")

    save_params(final, OUT / "classical_params.json")
    pd.DataFrame(log).to_csv(OUT / "classical_tuning.csv", index=False)
    print(f"\nsaved {OUT / 'classical_params.json'} and {OUT / 'classical_tuning.csv'}")


if __name__ == "__main__":
    main()
