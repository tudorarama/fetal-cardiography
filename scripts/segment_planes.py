"""Finds the heart and the thorax on the FETAL_PLANES_DB thorax images with the stage-1 U-Net.

Run from the project root (CPU, ~5 minutes for the 1,718 images):
    python scripts/segment_planes.py
    python scripts/segment_planes.py --device cuda      # only when the GPU is free

FETAL_PLANES_DB has no masks, so the U-Net trained on FOCUS (models/unet_pre.pth)
gives them: the same heart and thorax ellipses as for FOCUS, used by four_chambers/
to frame the annotation view, crop the heart for nnU-Net and compute the CTR.

The images come from other machines and are not all four-chamber views, so every
prediction gets quality measures and a verdict ("ok" = usable as a heart outline):
    conf_*     mean U-Net probability inside the predicted region (how sure it is)
    fit_*      overlap of the raw region with its fitted ellipse (is it ellipse-shaped)
    inside     share of the heart ellipse inside the thorax ellipse
    thorax_cut share of the thorax ellipse outside the image (zoomed-in images)
    ctr        heart area / thorax area (FOCUS: 0.2 - 0.45)
Outputs:
    data/ZENODO/masks/<image>-cardiac.png, <image>-thorax.png   (0 / 255, like FOCUS)
    results/planes_masks.csv                                       one row per image
    results/planes_masks_examples.png                              accepted / rejected examples
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import PLANES_DIR, STRUCTURES, ellipse_to_mask, load_planes_table
from src.measurements import ellipse_area
from src.postprocess import fill_holes, fit_ellipse, largest_component
from src.config import RESULTS_DIR as OUT
from src.preprocessing import prepare, restore_size
from src.unet import checkpoint_path, load_model

MODEL = checkpoint_path("pre")
MASK_DIR = PLANES_DIR / "masks"

LIMITS = {"conf": 0.75, "fit": 0.85, "inside": 0.95, "thorax_cut": 0.10, "ctr": (0.15, 0.55)}


def iou(a, b):
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def analyse(prob, shape):
    """prob: (2, H, W) probabilities at the original size -> masks, ellipses, measures."""
    row, masks, ellipses = {}, {}, {}
    for c, k in enumerate(STRUCTURES):
        raw = (prob[c] > 0.5).astype(np.uint8)
        clean = fill_holes(largest_component(raw))
        e = fit_ellipse(clean) if clean.any() else None
        tag = "heart" if k == "cardiac" else "thorax"
        if e is None:
            row.update({f"conf_{tag}": 0.0, f"fit_{tag}": 0.0})
            masks[k], ellipses[k] = np.zeros(shape, np.uint8), None
            continue
        m = ellipse_to_mask(shape, e)
        row[f"conf_{tag}"] = float(prob[c][clean > 0].mean())
        row[f"fit_{tag}"] = iou(clean, m)
        masks[k], ellipses[k] = m, e
    h, t = ellipses["cardiac"], ellipses["thorax"]
    if h is None or t is None:
        row.update({"inside": 0.0, "thorax_cut": 1.0, "ctr": np.nan})
    else:
        hm, tm = masks["cardiac"] > 0, masks["thorax"] > 0
        row["inside"] = float((hm & tm).sum() / max(hm.sum(), 1))
        row["thorax_cut"] = float(max(0.0, 1 - tm.sum() / ellipse_area(t)))
        row["ctr"] = float(ellipse_area(h) / ellipse_area(t))
    return masks, row


def verdict(r):
    reasons = []
    if min(r["conf_heart"], r["conf_thorax"]) < LIMITS["conf"]:
        reasons.append("unsure")
    if min(r["fit_heart"], r["fit_thorax"]) < LIMITS["fit"]:
        reasons.append("not elliptic")
    if r["inside"] < LIMITS["inside"]:
        reasons.append("heart outside thorax")
    if r["thorax_cut"] > LIMITS["thorax_cut"]:
        reasons.append("thorax cut off")
    lo, hi = LIMITS["ctr"]
    if not (lo <= r["ctr"] <= hi):
        reasons.append("implausible CTR")
    return ";".join(reasons)


def overlay(img, masks, size=260):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, col in (("thorax", (60, 200, 60)), ("cardiac", (40, 40, 230))):
        cnt, _ = cv2.findContours(masks[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnt, -1, col, max(2, img.shape[1] // 300))
    s = size / max(img.shape)
    return cv2.cvtColor(cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, help="only the first N images (quick test)")
    args = ap.parse_args()

    model, cfg = load_model(MODEL, args.device)
    table = load_planes_table()
    if args.limit:
        table = table.head(args.limit)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows, keep = [], {"ok": [], "rejected": []}
    rng = np.random.default_rng(0)
    for _, t in tqdm(table.iterrows(), total=len(table)):
        name = t["Image_name"]
        img = cv2.imread(str(t["image_path"]), cv2.IMREAD_GRAYSCALE)
        x, meta = prepare(img, cfg=cfg, return_meta=True)
        with torch.no_grad():
            p = torch.sigmoid(model(x[None].to(args.device)))[0].cpu().numpy()
        prob = np.stack([restore_size(p[c], meta["resize"], interp=cv2.INTER_LINEAR)
                         for c in range(len(STRUCTURES))])
        masks, row = analyse(prob, img.shape)
        for k in STRUCTURES:
            cv2.imwrite(str(MASK_DIR / f"{name}-{k}.png"), masks[k] * 255)
        row["reasons"] = verdict(row)
        row["ok"] = row["reasons"] == ""
        rows.append({"name": name, "patient": int(t["Patient_num"]), "official_train": int(t["Train"]),
                     "machine": t["US_Machine"], **row})
        bucket = keep["ok" if row["ok"] else "rejected"]
        if len(bucket) < 8 or rng.random() < 8 / len(rows):  # random examples of both kinds
            if len(bucket) >= 8:
                bucket.pop(rng.integers(len(bucket)))
            bucket.append((name, row["reasons"] or f"CTR {row['ctr']:.2f}", overlay(img, masks)))

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "planes_masks.csv", index=False)
    n_ok = int(df["ok"].sum())
    print(f"\n{n_ok} / {len(df)} images with a usable heart + thorax ({df[df.ok].patient.nunique()} patients)")
    reasons = df.loc[~df.ok, "reasons"].str.split(";").explode().value_counts()
    print("rejected because:\n" + reasons.to_string())
    print(df.groupby("machine")["ok"].agg(["sum", "count"]).to_string())

    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for r, kind in enumerate(("ok", "ok", "rejected", "rejected")):
        items = keep[kind][(r % 2) * 4:(r % 2) * 4 + 4]
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(items):
                name, label, vis = items[c]
                ax.imshow(vis)
                ax.set_title(f"{kind.upper()} {name[:18]}\n{label}", fontsize=9)
    fig.suptitle("U-Net (stage 1) on FETAL_PLANES_DB: heart red, thorax green", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "planes_masks_examples.png", dpi=80)
    print(f"saved {OUT / 'planes_masks.csv'}, planes_masks_examples.png and masks in {MASK_DIR}")


if __name__ == "__main__":
    main()
