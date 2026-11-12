"""Generate a before/after grid comparing raw and preprocessed ultrasound images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import list_cases, load_case, load_image, load_planes_table, overlay
from src.preprocessing import preprocess


def run_focus(split, n):
    names = list_cases(split)[:n]
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        img, _, masks = load_case(split, name)
        proc = preprocess(img)
        row[0].imshow(img, cmap="gray"); row[0].set_title(f"{name} - raw")
        row[1].imshow(proc, cmap="gray"); row[1].set_title("preprocessed")
        row[2].imshow(overlay(proc, masks)); row[2].set_title("preprocessed + GT")
        for ax in row:
            ax.axis("off")
    return fig, "preprocessing_demo.png"


def run_zenodo(split, n):
    df = load_planes_table()
    df = df[df["Train"] == (1 if split == "train" else 0)].head(n)
    fig, axes = plt.subplots(len(df), 2, figsize=(8, 4 * len(df)), squeeze=False)
    for row, (_, item) in zip(axes, df.iterrows()):
        img = load_image(item["image_path"])
        proc = preprocess(img)
        row[0].imshow(img, cmap="gray"); row[0].set_title(f"{item['Image_name']} - raw")
        row[1].imshow(proc, cmap="gray"); row[1].set_title("preprocessed")
        for ax in row:
            ax.axis("off")
    return fig, "preprocessing_zenodo_demo.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="focus", choices=["focus", "zenodo"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    if args.dataset == "focus":
        fig, filename = run_focus(args.split, args.n)
    else:
        fig, filename = run_zenodo(args.split, args.n)

    out = Path(__file__).resolve().parent.parent / "results" / filename
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()