"""Evaluates trained U-Nets and compares them with the classical method.

Run from the project root (after train_unet.py and evaluate_classical.py):
    python scripts/evaluate_unet.py                 # test set, every checkpoint found
    python scripts/evaluate_unet.py --split val

For each checkpoint results/unet_<name>.pth it writes results/unet_<name>_<split>.csv,
then prints a comparison table (classical, prior, U-Nets) and a paired Wilcoxon
signed-rank test on the per-image heart and thorax Dice, U-Net vs classical.
Figures: results/unet_<name>_<split>_examples.png and results/comparison_<split>_dice.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from src.data import load_case, overlay
from src.evaluation import evaluate, summarize
from src.unet import make_predictor

OUT = ROOT / "results"


def examples_figure(df, split, predict, path, title, n=3):
    order = df.sort_values("dice_cardiac")
    picks = list(order.tail(n)["name"])[::-1] + list(order.head(n)["name"])
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    for ax, name in zip(axes.flat, picks):
        img, _, gt = load_case(split, name)
        masks, _ = predict(img)
        vis = overlay(img, gt, thickness=3)
        vis = overlay(vis, masks, colors={"thorax": (255, 255, 0), "cardiac": (255, 0, 255)})
        row = df[df["name"] == name].iloc[0]
        ax.imshow(vis)
        ax.set_title(f"{name}: Dice heart {row.dice_cardiac:.2f}, thorax {row.dice_thorax:.2f}")
        ax.axis("off")
    fig.suptitle(f"{title}: best (top) and worst (bottom). GT green/red, prediction yellow/magenta")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dice_boxplot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, k, label in zip(axes, ("dice_cardiac", "dice_thorax"), ("heart", "thorax")):
        ax.boxplot([df[k] for df in results.values()], tick_labels=list(results), showmeans=True)
        ax.set(title=f"Dice, {label}", ylim=(0, 1.02))
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    results = {}
    for name in ("classical", "prior"):
        csv = OUT / f"{name}_{args.split}.csv"
        if csv.exists():
            # names like "001" must stay strings, otherwise they do not match the U-Net rows
            results[name] = pd.read_csv(csv, dtype={"name": str})

    # unet_<name>_last.pth are resume states (last epoch), not the selected models
    ckpts = sorted(c for c in OUT.glob("unet_*.pth") if not c.stem.endswith("_last"))
    if not ckpts:
        sys.exit("no checkpoint found: run scripts/train_unet.py first")
    for ckpt in ckpts:
        tag = ckpt.stem  # unet_pre / unet_raw
        predict = make_predictor(ckpt)
        df = evaluate(predict, args.split, progress=True)
        df.to_csv(OUT / f"{tag}_{args.split}.csv", index=False)
        results[tag] = df
        examples_figure(df, args.split, predict, OUT / f"{tag}_{args.split}_examples.png", tag)

    table = pd.DataFrame({k: summarize(v)["mean"] for k, v in results.items()})
    print(f"\n=== {args.split} set (mean over images)")
    print(table.round(3).to_string())
    ctr_alarm = {k: int((v["ctr_pred"] > 0.35).sum()) for k, v in results.items()}
    print("images with predicted CTR > 0.35:", ctr_alarm,
          "| ground truth:", int((next(iter(results.values()))["ctr_gt"] > 0.35).sum()))

    if "classical" in results:
        print("\nWilcoxon signed-rank test vs classical (paired, per image):")
        base = results["classical"].set_index("name")
        for tag in (t for t in results if t.startswith("unet_")):
            other = results[tag].set_index("name").loc[base.index]
            for k in ("dice_cardiac", "dice_thorax"):
                p = wilcoxon(other[k], base[k]).pvalue
                print(f"  {tag} {k}: median diff {(other[k] - base[k]).median():+.3f}, p = {p:.2g}")

    dice_boxplot(results, OUT / f"comparison_{args.split}_dice.png")
    print(f"\nresults saved to {OUT}")


if __name__ == "__main__":
    main()
