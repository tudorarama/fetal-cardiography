"""Figures for the README and the presentation.

Run from the project root (after scripts/validate_pipeline.py):
    python scripts/make_figures.py

Saves to results/:
  preprocessing.png   original -> median -> bilateral -> CLAHE on two images
  ctr_histogram.png   distribution of the ground-truth CTR on FOCUS
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_case
from src.preprocess import preprocess_steps

OUT = ROOT / "results"


def preprocessing_figure(cases=(("train", "001"), ("train", "003"))):
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3.4 * len(cases)))
    for row, (split, name) in zip(axes, cases):
        img, _, _ = load_case(split, name)
        for ax, (title, im) in zip(row, preprocess_steps(img)):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}: {title}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "preprocessing.png", dpi=130)
    plt.close(fig)


def ctr_histogram(threshold=0.35):
    csv = OUT / "gt_measurements.csv"
    if not csv.exists():
        sys.exit("run scripts/validate_pipeline.py first")
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["ctr_area_gt"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].axvline(threshold, color="#C44E52", ls="--", label=f"threshold {threshold}")
    axes[0].set(title="CTR (area ratio), FOCUS ground truth", xlabel="CTR", ylabel="images")
    axes[0].legend()
    axes[1].hist(df["ctr_circ_gt"], bins=30, color="#55A868", edgecolor="white")
    axes[1].set(title="CTR (circumference ratio), FOCUS ground truth", xlabel="CTR")
    fig.tight_layout()
    fig.savefig(OUT / "ctr_histogram.png", dpi=130)
    plt.close(fig)
    print(f"cases above {threshold}: {(df['ctr_area_gt'] > threshold).sum()} / {len(df)}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    preprocessing_figure()
    ctr_histogram()
    print(f"figures saved to {OUT}")
