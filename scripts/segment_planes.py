"""Finds the heart and the thorax on the FETAL_PLANES_DB thorax images with the stage-1 U-Net.

Run from the project root (CPU, ~5 minutes for the 1,718 images):
    python scripts/segment_planes.py
    python scripts/segment_planes.py --device cuda      # only when the GPU is free

FETAL_PLANES_DB has no masks, so the U-Net trained on FOCUS (models/unet_pre.pth)
gives them: the same heart and thorax ellipses as for FOCUS, used by four_chambers/
to frame the annotation view, crop the heart for nnU-Net and compute the CTR.

Every prediction gets the quality measures of src.outline and a verdict
("ok" = usable as a heart outline).
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
from tqdm import tqdm

from src.config import RESULTS_DIR as OUT
from src.data import PLANES_DIR, STRUCTURES, load_planes_table
from src.outline import analyse, draw, predict_probs, verdict
from src.unet import checkpoint_path, load_model

MODEL = checkpoint_path("pre")
MASK_DIR = PLANES_DIR / "masks"

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
        masks, row = analyse(predict_probs(model, cfg, img, args.device), img.shape)
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
            bucket.append((name, row["reasons"] or f"CTR {row['ctr']:.2f}", draw(img, masks)))

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
