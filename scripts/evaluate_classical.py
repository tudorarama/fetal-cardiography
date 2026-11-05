"""Evaluates the classical method on the test set (run tune_classical.py first).

Run from the project root:
    python scripts/evaluate_classical.py            # test set
    python scripts/evaluate_classical.py --split val

Also evaluates a trivial "prior" baseline - a fixed thorax circle at the mean
position and size of the training set, and a heart circle of the mean CTR at
the mean heart offset - so it is clear how much the image processing adds.

Outputs in results/:
  classical_<split>.csv, prior_<split>.csv   per-case metrics
  classical_<split>_summary.csv              mean / std / median
  classical_<split>_examples.png             best and worst cases
  classical_<split>_ctr.png                  predicted vs ground-truth CTR
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classical import load_params, make_predictor
from src.data import ellipse_to_mask, list_cases, load_case, overlay
from src.evaluation import evaluate, summarize

OUT = ROOT / "results"


def make_prior_predictor():
    """Fixed circles learnt from the training ground truth (no image processing)."""
    rel = []
    for name in list_cases("train"):
        img, ell, _ = load_case("train", name)
        h, w = img.shape
        tx, ty, ta, tb, _ = ell["thorax"]
        hx, hy, ha, hb, _ = ell["cardiac"]
        r = np.sqrt(ta * tb)
        rel.append((tx / w, ty / h, r / min(h, w), (hx - tx) / r, (hy - ty) / r,
                    np.sqrt(ha * hb) / r))
    mx, my, mr, hdx, hdy, hr = np.mean(rel, axis=0)

    def predict(img):
        h, w = img.shape
        r = mr * min(h, w)
        t = (mx * w, my * h, r, r, 0.0)
        c = (t[0] + hdx * r, t[1] + hdy * r, hr * r, hr * r, 0.0)
        return ({"thorax": ellipse_to_mask(img.shape, t), "cardiac": ellipse_to_mask(img.shape, c)},
                {"thorax": t, "cardiac": c})

    return predict


def examples_figure(df, split, p, path, n=3):
    order = df.sort_values("dice_cardiac")
    picks = list(order.tail(n)["name"])[::-1] + list(order.head(n)["name"])
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    predict = make_predictor(p)
    for ax, name in zip(axes.flat, picks):
        img, _, gt = load_case(split, name)
        masks, _ = predict(img)
        vis = overlay(img, gt, thickness=3)
        vis = overlay(vis, masks, colors={"thorax": (255, 255, 0), "cardiac": (255, 0, 255)})
        row = df[df["name"] == name].iloc[0]
        ax.imshow(vis)
        ax.set_title(f"{name}: Dice heart {row.dice_cardiac:.2f}, thorax {row.dice_thorax:.2f}")
        ax.axis("off")
    fig.suptitle("Best (top) and worst (bottom) cases. GT: green thorax / red heart; "
                 "prediction: yellow thorax / magenta heart")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def ctr_figure(df, path):
    d = df.dropna(subset=["ctr_pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d["ctr_gt"], d["ctr_pred"], s=18, alpha=0.7)
    lim = [0.1, max(0.5, d["ctr_pred"].max() + 0.02)]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(0.35, color="#C44E52", ls=":", label="threshold 0.35")
    ax.set(xlim=lim, ylim=lim, xlabel="CTR ground truth", ylabel="CTR classical",
           title="Cardiothoracic ratio (area)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    params_path = OUT / "classical_params.json"
    if not params_path.exists():
        sys.exit("run scripts/tune_classical.py first")
    p = load_params(params_path)

    df = evaluate(make_predictor(p), args.split, progress=True)
    prior = evaluate(make_prior_predictor(), args.split)
    df.to_csv(OUT / f"classical_{args.split}.csv", index=False)
    prior.to_csv(OUT / f"prior_{args.split}.csv", index=False)

    summary = summarize(df)
    summary.to_csv(OUT / f"classical_{args.split}_summary.csv")
    table = pd.DataFrame({"classical": summary["mean"], "prior (fixed circles)": summarize(prior)["mean"]})
    print(f"\n=== {args.split} set, {len(df)} images (mean)")
    print(table.round(3).to_string())

    examples_figure(df, args.split, p, OUT / f"classical_{args.split}_examples.png")
    ctr_figure(df, OUT / f"classical_{args.split}_ctr.png")
    print(f"\nresults saved to {OUT}")


if __name__ == "__main__":
    main()
