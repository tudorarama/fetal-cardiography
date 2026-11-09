"""Predicts the four chambers with the trained nnU-Net.

Run from the four_chambers folder (after train.py):
    python predict.py              # annotated test images      -> nnunet/predictions/test
    python predict.py --prefill    # all train/val not annotated -> nnunet/predictions/prefill
                                   #   then: python annotate.py --prefill

Like in training, every image is cropped to the square around the FOCUS heart
outline and resized to 384 x 384. nnU-Net returns class probabilities for the
crop; they are resized back to the original crop size (bilinear), placed into
the full image and only then turned into labels (argmax), which gives smoother
contours than resizing the labels. Output: one label map per image at the
original resolution (PNG, 0 background, 1 LV, 2 RV, 3 LA, 4 RA).

Images outside FOCUS have no heart outline; for them the heart would first have
to be found (e.g. with the heart/thorax U-Net of the first project).
By default the final checkpoint is used, as nnU-Net does (--best: checkpoint_best.pth).
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import (ROI_SIZE, case_id, completed, crop, heart_roi_box, is_complete,
                    list_cases, load_annotation, load_image, uncrop)
from config import (ANN_DIR, CONFIGURATION, DATASET_ID, PREDICTIONS_DIR, setup_nnunet_env)
from train import LAST_TRAINER, run, tool


def cases_to_predict(prefill):
    """Test: the annotated test images. Prefill: EVERY train/val FOCUS image without a
    finished annotation (so any image chosen in the annotation tool's picker is pre-drawn).
    Test images are never pre-filled: the test set must be drawn by hand."""
    if not prefill:
        return [c for c in completed() if c[0] == "test"]
    out = []
    for split in ("train", "val"):
        for name in list_cases(split):
            ann = load_annotation(split, name)
            if ann is None or not (is_complete(ann) or ann.get("skipped")):
                out.append((split, name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--trainer", help="default: the one used by the last train.py run")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--best", action="store_true", help="use checkpoint_best.pth")
    ap.add_argument("--no-tta", action="store_true", help="disable mirroring test-time augmentation")
    args = ap.parse_args()
    setup_nnunet_env()

    trainer = args.trainer or (LAST_TRAINER.read_text().strip() if LAST_TRAINER.exists() else None)
    if trainer is None:
        raise SystemExit("no trained model: run train.py first (or pass --trainer)")
    tag = "prefill" if args.prefill else "test"
    cases = cases_to_predict(args.prefill)
    if not cases:
        raise SystemExit("nothing to predict")
    roi_in, roi_out, final = (PREDICTIONS_DIR / f"{tag}_roi_input", PREDICTIONS_DIR / f"{tag}_roi",
                              PREDICTIONS_DIR / tag)
    for d in (roi_in, roi_out, final):
        if d.exists():
            shutil.rmtree(d)  # regenerated every run
        d.mkdir(parents=True)

    boxes, shapes = {}, {}
    for split, name in cases:
        img = load_image(split, name)
        cid = case_id(split, name)
        boxes[cid], shapes[cid] = heart_roi_box(split, name), img.shape
        cv2.imwrite(str(roi_in / f"{cid}_0000.png"), crop(img, boxes[cid], ROI_SIZE, cv2.INTER_LINEAR))
    print(f"{len(cases)} images ({tag})")

    cmd = [tool("nnUNetv2_predict"), "-i", str(roi_in), "-o", str(roi_out), "-d", str(DATASET_ID),
           "-c", CONFIGURATION, "-f", args.fold, "-tr", trainer, "--save_probabilities",
           "-chk", "checkpoint_best.pth" if args.best else "checkpoint_final.pth"]
    if args.no_tta:
        cmd.append("--disable_tta")
    run(cmd)

    for cid, box in boxes.items():
        prob = np.load(roi_out / f"{cid}.npz")["probabilities"].astype(np.float32)
        if np.isnan(prob).any():  # nnU-Net predicts in fp16; GTX 16xx cards can produce NaN
            raise SystemExit(f"{cid}: NaN in the prediction (fp16 on this GPU?)")
        prob = prob.reshape(prob.shape[0], ROI_SIZE, ROI_SIZE).transpose(1, 2, 0)  # (H, W, C)
        full = uncrop(np.ascontiguousarray(prob), box, shapes[cid])  # zeros outside -> background
        cv2.imwrite(str(final / f"{cid}.png"), full.argmax(-1).astype(np.uint8))
    print(f"predictions (original resolution) in {final}")


if __name__ == "__main__":
    main()
