"""Inspect dataset splits: image counts, shapes, intensity stats, and background ratio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import list_cases, load_case, load_image, load_planes_table


def summarize_images(label, images):
    shapes = set()
    total_pixels = 0
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    black_pixels = 0

    for img in images:
        shapes.add(img.shape)
        arr = img.astype(np.float64)
        total_pixels += arr.size
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr ** 2).sum()
        black_pixels += int((img == 0).sum())

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
    black_pct = 100.0 * black_pixels / total_pixels
    shape_info = "varying" if len(shapes) > 1 else "fixed"

    print(f"{label}")
    print(f"  images    : {len(images)}")
    print(f"  shapes    : {sorted(shapes)} ({shape_info})")
    print(f"  intensity : mean={mean:.2f}, std={std:.2f} (norm: mean={mean/255:.4f}, std={std/255:.4f})")
    print(f"  background: {black_pct:.1f}% black pixels")
    return mean, std


def main():
    print("--- FOCUS ---")
    train_imgs = [load_case("train", n)[0] for n in list_cases("train")]
    val_imgs = [load_case("val", n)[0] for n in list_cases("val")]
    test_imgs = [load_case("test", n)[0] for n in list_cases("test")]

    train_mean, train_std = summarize_images("train", train_imgs)
    summarize_images("val", val_imgs)
    summarize_images("test", test_imgs)
    print(f"train normalization: mean={train_mean:.2f}, std={train_std:.2f}\n")

    print("--- ZENODO (FETAL_PLANES_DB) ---")
    df = load_planes_table()
    z_train = [load_image(p) for p in df[df["Train"] == 1]["image_path"]]
    z_test = [load_image(p) for p in df[df["Train"] == 0]["image_path"]]
    summarize_images("train", z_train)
    summarize_images("test", z_test)


if __name__ == "__main__":
    main()