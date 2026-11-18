"""Finds the heart and the thorax on the Heartbeat four-chamber images with the stage-1 U-Nets.

Run from the project root (CPU, ~10 minutes for the 3,283 4C images):
    python scripts/segment_heartbeat.py
    python scripts/segment_heartbeat.py --limit 20     # quick test

Heartbeat (data/HEARTBEAT/, restricted access) only says CHD / No_CHD per patient.
This is step 1 of the anomaly test on it:
    1. here: heart outline (crop for nnU-Net) and CTR of every 4C image
    2. four_chambers/predict_heartbeat.py: chambers with nnU-Net v2
    3. scripts/evaluate_heartbeat.py: anomaly rules against the CHD label

Two U-Nets are used:
    unet_pre      heart outline for the nnU-Net crop (the one used for FETAL_PLANES_DB)
    unet_pre_cm   CTR, because it was trained with random heart sizes and is the
                  only one that does not underestimate enlarged hearts
Images are left out when:
    doppler       colour Doppler (> 1 % coloured pixels): the U-Nets saw only B-mode
    heart_ok = 0  the heart is not found clearly -> no chambers
    ctr_ok = 0    heart and thorax not usable together (src.outline.verdict) -> no CTR
Outputs (local only: the data may not be redistributed):
    data/HEARTBEAT_masks/<id>-cardiac.png, <id>-thorax.png   (0 / 255, like FOCUS)
    results/heartbeat_outline.csv                             one row per image
    results/heartbeat_outline_examples.png                    accepted / rejected examples
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
from tqdm import tqdm

from src.config import DATA_DIR, RESULTS_DIR as OUT
from src.data import STRUCTURES, load_heartbeat_table
from src.outline import analyse, draw, heart_found, predict_probs, verdict
from src.unet import checkpoint_path, load_model

MASK_DIR = DATA_DIR / "HEARTBEAT_masks"
DOPPLER_FRACTION = 0.01


def colour_fraction(bgr):
    """Share of clearly coloured pixels (colour Doppler); B-mode images are grey."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return float(((np.abs(b - g) + np.abs(g - r)) > 40).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, help="only the first N images (quick test)")
    args = ap.parse_args()

    outline_model, outline_cfg = load_model(checkpoint_path("pre"), args.device)
    ctr_model, ctr_cfg = load_model(checkpoint_path("pre_cm"), args.device)
    table = load_heartbeat_table("4C")
    if args.limit:
        table = table.sample(args.limit, random_state=0)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    rows, keep = [], {"ok": [], "rejected": []}
    rng = np.random.default_rng(0)
    for _, t in tqdm(table.iterrows(), total=len(table)):
        bgr = cv2.imread(str(t["image_path"]))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        row = {k: t[k] for k in ("id", "exam", "patient", "trimester", "part", "chd")}
        row["colour"] = colour_fraction(bgr)
        row["doppler"] = row["colour"] > DOPPLER_FRACTION

        masks, q = analyse(predict_probs(outline_model, outline_cfg, img, args.device), img.shape)
        row.update(q)
        row["heart_ok"] = (not row["doppler"]) and heart_found(q)
        for k in STRUCTURES:
            cv2.imwrite(str(MASK_DIR / f"{t['id']}-{k}.png"), masks[k] * 255)

        _, q_cm = analyse(predict_probs(ctr_model, ctr_cfg, img, args.device), img.shape)
        reasons = verdict(q_cm)
        row["ctr_cm"] = q_cm["ctr"]
        row["ctr_reasons"] = "doppler" if row["doppler"] else reasons
        row["ctr_ok"] = row["ctr_reasons"] == ""
        rows.append(row)

        kind = "ok" if row["ctr_ok"] else "rejected"
        bucket = keep[kind]
        if len(bucket) < 8 or rng.random() < 8 / len(rows):
            if len(bucket) >= 8:
                bucket.pop(rng.integers(len(bucket)))
            label = row["ctr_reasons"] or f"CTR {row['ctr_cm']:.2f}"
            bucket.append((t["id"], label, draw(img, masks)))

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "heartbeat_outline.csv", index=False)
    print(f"\n{len(df)} images: {int(df.doppler.sum())} colour Doppler, "
          f"{int(df.heart_ok.sum())} with a clear heart (chambers), {int(df.ctr_ok.sum())} with a usable CTR")
    reasons = df.loc[~df.ctr_ok, "ctr_reasons"].str.split(";").explode().value_counts()
    print("no CTR because:\n" + reasons.to_string())
    print(df.groupby("chd")[["heart_ok", "ctr_ok"]].mean().round(3).to_string())

    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for r, kind in enumerate(("ok", "ok", "rejected", "rejected")):
        items = keep[kind][(r % 2) * 4:(r % 2) * 4 + 4]
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(items):
                name, label, vis = items[c]
                ax.imshow(vis)
                ax.set_title(f"{kind.upper()} {name}\n{label}", fontsize=9)
    fig.suptitle("U-Net (stage 1) on Heartbeat 4C: heart red, thorax green (local only, do not share)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "heartbeat_outline_examples.png", dpi=80)
    print(f"saved {OUT / 'heartbeat_outline.csv'}, heartbeat_outline_examples.png and masks in {MASK_DIR}")


if __name__ == "__main__":
    main()
