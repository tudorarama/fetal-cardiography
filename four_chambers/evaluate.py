"""Evaluates the predicted chambers against the annotated TEST images.

Run from the four_chambers folder (after predict.py):
    python evaluate.py
    python evaluate.py --pred nnunet/predictions/test --name final

Per image and chamber: Dice and HD95; per image: LV/RV, LA/RA (annotation vs
prediction), cardiac axis (only where a spine/aorta landmark was annotated) and
the anomaly flags of measurements.screen().
Test images of both sources (FOCUS test, FETAL_PLANES_DB ptest) are evaluated;
the summary is also given per source (FETAL_PLANES_DB = other machines).
Outputs: results/evaluation_<name>.csv, results/examples_<name>.png
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (case_id, clahe, completed, landmark, load_annotation, load_focus_masks,
                    load_image, rasterize, role, source)
from config import CHAMBERS, LABELS, PREDICTIONS_DIR, RESULTS_DIR
from measurements import chamber_measurements, dice, hd95, screen

RGB = {"LV": (230, 60, 60), "RV": (40, 120, 255), "LA": (250, 200, 30), "RA": (60, 200, 60)}


def colorize(img, lab, alpha=0.45):
    vis = np.stack([img] * 3, -1).astype(np.float32)
    for c in CHAMBERS:
        m = lab == LABELS[c]
        vis[m] = (1 - alpha) * vis[m] + alpha * np.array(RGB[c])
    return vis.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PREDICTIONS_DIR / "test"))
    ap.add_argument("--name", default="test")
    args = ap.parse_args()

    cases = [c for c in completed() if role(c[0]) == "test"]
    if not cases:
        raise SystemExit("no annotated test images")
    rows, vis = [], []
    for split, name in cases:
        cid = case_id(split, name)
        img = load_image(split, name)
        gt = rasterize(load_annotation(split, name), img.shape)  # full resolution annotation
        pred = cv2.imread(str(f"{args.pred}/{cid}.png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise SystemExit(f"missing prediction {args.pred}/{cid}.png: run predict.py")
        row = {"case": cid, "source": source(split)}
        for c in CHAMBERS:
            p, g = pred == LABELS[c], gt == LABELS[c]
            row[f"dice_{c}"], row[f"hd95_{c}"] = dice(p, g), hd95(p, g)
        row["dice_mean"] = float(np.mean([row[f"dice_{c}"] for c in CHAMBERS]))
        lm = landmark(load_annotation(split, name))
        ys, xs = np.nonzero(load_focus_masks(split, name)["thorax"])
        center = (xs.mean(), ys.mean())
        for tag, lab in (("gt", gt), ("pred", pred)):
            m = chamber_measurements(lab, lm[:2] if lm else None, center)
            row.update({f"lv_rv_{tag}": m["lv_rv"], f"la_ra_{tag}": m["la_ra"],
                        f"axis_{tag}": m["axis"], f"flags_{tag}": ";".join(screen(m))})
        rows.append(row)
        vis.append((cid, img, gt, pred, row["dice_mean"]))

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"evaluation_{args.name}.csv", index=False)

    print(f"=== {len(df)} annotated test images")
    print(f"{'':6s} {'Dice mean':>9s} {'median':>7s} {'min':>6s} {'HD95 px':>8s}")
    for c in CHAMBERS:
        d = df[f"dice_{c}"]
        print(f"{c:6s} {d.mean():9.3f} {d.median():7.3f} {d.min():6.3f} {df[f'hd95_{c}'].mean():8.1f}")
    print(f"{'all':6s} {df['dice_mean'].mean():9.3f} {df['dice_mean'].median():7.3f} "
          f"{df['dice_mean'].min():6.3f}")
    for k in ("lv_rv", "la_ra", "axis"):
        g, p = df[f"{k}_gt"].astype(float), df[f"{k}_pred"].astype(float)
        if g.notna().any():
            print(f"{k:6s} annotation {g.mean():.2f} | prediction {p.mean():.2f} | "
                  f"mean abs error {(p - g).abs().mean():.2f} (n={int(g.notna().sum())})")
    same = (df["flags_gt"].fillna("") == df["flags_pred"].fillna("")).mean()
    print(f"same anomaly flags as the annotation: {same:.0%}")
    if df["source"].nunique() > 1:
        print("per source (mean Dice):")
        for src, g in df.groupby("source"):
            print(f"  {src:16s} n={len(g):3d}  "
                  + "  ".join(f"{c} {g[f'dice_{c}'].mean():.3f}" for c in CHAMBERS)
                  + f"  all {g['dice_mean'].mean():.3f}")

    n = len(vis)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.6 * n), squeeze=False)
    for r, (cid, img, gt, pred, dm) in enumerate(sorted(vis, key=lambda v: -v[4])):
        ys, xs = np.nonzero((gt > 0) | (pred > 0))
        pad = 60
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        view = clahe(img)
        for ax, im, title in zip(axes[r], (view, colorize(view, gt), colorize(view, pred)),
                                 (cid, "annotation", f"nnU-Net, mean Dice {dm:.2f}")):
            ax.imshow(im[y0:y1, x0:x1], cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle("LV red, RV blue, LA yellow, RA green (best images first)", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))  # keep room for the title
    fig.savefig(RESULTS_DIR / f"examples_{args.name}.png", dpi=90)
    plt.close(fig)
    print(f"saved {RESULTS_DIR / f'evaluation_{args.name}.csv'} and examples_{args.name}.png")


if __name__ == "__main__":
    main()
