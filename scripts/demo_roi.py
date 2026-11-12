"""Visualize cardiac ROI cropping and full-frame mask reconstruction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import list_cases, load_case
from src.preprocessing import crop_heart_roi, paste_back, preprocess


def main():
    names = list_cases("train")[:4]
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)), squeeze=False)

    for row, name in zip(axes, names):
        img, _, masks = load_case("train", name)
        proc = preprocess(img)
        crop, roi_meta = crop_heart_roi(proc, masks["cardiac"], margin=0.25)

        x0, y0, x1, y1 = roi_meta["box"]
        boxed = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(boxed, (x0, y0), (x1, y1), (0, 255, 0), 2)

        reconstructed = paste_back(masks["cardiac"][y0:y1, x0:x1], roi_meta)

        row[0].imshow(boxed); row[0].set_title(f"{name} - ROI box")
        row[1].imshow(crop, cmap="gray"); row[1].set_title("cropped heart ROI")
        row[2].imshow(reconstructed, cmap="gray"); row[2].set_title("paste_back mask")
        for ax in row:
            ax.axis("off")

    out = Path(__file__).resolve().parent.parent / "results" / "roi_demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()