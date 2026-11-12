"""Verify cached dataset integrity, uniform dimensions, and mask-to-image overlay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import UNET_SIZE
from src.data import list_cases


def verify_split(split, proc_root):
    expected = list_cases(split)
    split_dir = proc_root / split
    assert split_dir.exists(), f"Missing directory: {split_dir}"

    npy_files = sorted(split_dir.glob("*.npy"))
    assert len(npy_files) == len(expected), (
        f"{split}: expected {len(expected)} images, found {len(npy_files)}"
    )

    for name in expected:
        img_path = split_dir / f"{name}.npy"
        mask_path = split_dir / f"{name}_masks.npz"
        meta_path = split_dir / f"{name}_meta.json"

        assert img_path.exists(), f"Missing image: {img_path}"
        assert mask_path.exists(), f"Missing masks: {mask_path}"
        assert meta_path.exists(), f"Missing meta: {meta_path}"

        arr = np.load(img_path)
        assert arr.shape == (1, UNET_SIZE, UNET_SIZE), f"{img_path}: bad shape {arr.shape}"
        assert arr.dtype == np.float32, f"{img_path}: bad dtype {arr.dtype}"
        assert np.isfinite(arr).all(), f"{img_path}: contains NaN or Inf"

        with np.load(mask_path) as masks:
            assert len(masks.files) > 0, f"{mask_path}: empty mask archive"
            for m_name in masks.files:
                m = masks[m_name]
                assert m.shape == (UNET_SIZE, UNET_SIZE), (
                    f"{mask_path}[{m_name}]: bad shape {m.shape}"
                )
                assert set(np.unique(m)).issubset({0, 1}), (
                    f"{mask_path}[{m_name}]: non-binary values"
                )

    print(
        f"ok  {split}: {len(expected)} cases verified | "
        f"image=(1, {UNET_SIZE}, {UNET_SIZE}) | mask=({UNET_SIZE}, {UNET_SIZE})"
    )


def _make_overlay(arr, masks):
    img = arr[0]
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    rgb = np.stack([img_norm] * 3, axis=-1)

    other_keys = [k for k in masks.files if k != "cardiac"]
    if other_keys:
        union = np.zeros((UNET_SIZE, UNET_SIZE), dtype=bool)
        for k in other_keys:
            union |= masks[k] > 0
        rgb[union, 0] = np.clip(rgb[union, 0] * 0.5 + 0.5, 0.0, 1.0)
        rgb[union, 1] *= 0.5
        rgb[union, 2] *= 0.5

    if "cardiac" in masks.files:
        cardiac = (masks["cardiac"] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(cardiac, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_u8 = (rgb * 255).astype(np.uint8)
        cv2.drawContours(rgb_u8, contours, -1, (0, 255, 0), 1)
        rgb = rgb_u8.astype(np.float32) / 255.0

    return rgb


def save_overlay_grid(splits, proc_root, out_path, samples_per_split=3):
    rows = len(splits)
    cols = samples_per_split
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for r, split in enumerate(splits):
        names = list_cases(split)[:samples_per_split]
        for c in range(cols):
            ax = axes[r][c]
            if c >= len(names):
                ax.axis("off")
                continue
            name = names[c]
            arr = np.load(proc_root / split / f"{name}.npy")
            with np.load(proc_root / split / f"{name}_masks.npz") as masks:
                overlay = _make_overlay(arr, masks)
            ax.imshow(overlay)
            ax.set_title(f"{split} / {name}")
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved overlay: {out_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    proc_root = root / "data" / "processed"
    splits = ["train", "val", "test"]

    for split in splits:
        verify_split(split, proc_root)

    save_overlay_grid(splits, proc_root, root / "results" / "processed_overlay.png")


if __name__ == "__main__":
    main()