"""Exports the completed annotations to the nnU-Net v2 raw dataset format.

Run from the four_chambers folder (train.py also runs it automatically):
    python prepare_dataset.py

Output: nnunet/raw/Dataset501_FetalHeart4CH/
    imagesTr/train_106_0000.png   384 x 384 crop around the heart (train + val annotations,
                                  FETAL_PLANES_DB as ptrain_<image>_0000.png)
    labelsTr/train_106.png        label map 0 background, 1 LV, 2 RV, 3 LA, 4 RA (same crop)
    imagesTs/, labelsTs/          the annotated TEST images (never used for training)
    dataset.json                  description required by nnU-Net
    splits_final.json             fold 0: train + ptrain -> training, val + pval -> validation
    crops.json                    crop box of every case, to map predictions back

Region of interest: a square around the heart outline (common.heart_roi_box),
resized to 384 x 384. The chambers are all inside the heart; the full 959 x 661
image made one epoch take ~14 minutes on the GTX 1660.

The folder is regenerated from annotations/ every time (it is derived data).
nnU-Net normally makes a 5-fold cross-validation; here fold 0 follows the
official FOCUS split, so the validation images are the same as in the first
project and the test images stay untouched until the final evaluation.
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import json
import shutil

import cv2

from common import (ROI_SIZE, case_id, completed, crop, heart_roi_box, load_annotation,
                    load_image, rasterize, role)
from config import CHAMBERS, DATASET_NAME, LABELS, NNUNET_DIR, setup_nnunet_env


def raw_dir():
    return NNUNET_DIR / "raw" / DATASET_NAME


def export():
    setup_nnunet_env()
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    cases = completed()
    groups = {"Tr": [c for c in cases if role(c[0]) in ("train", "val")],
              "Ts": [c for c in cases if role(c[0]) == "test"]}
    n_val = sum(1 for s, _ in groups["Tr"] if role(s) == "val")
    if len(groups["Tr"]) - n_val < 2 or n_val < 1:
        raise SystemExit(f"not enough annotations: {len(groups['Tr']) - n_val} train, {n_val} val "
                         "(need at least 2 train and 1 val) - run annotate.py")

    out = raw_dir()
    if out.exists():
        shutil.rmtree(out)  # derived data, rebuilt from annotations/
    boxes = {}
    for g, items in groups.items():
        (out / f"images{g}").mkdir(parents=True)
        (out / f"labels{g}").mkdir(parents=True)
        for split, name in items:
            img = load_image(split, name)
            lab = rasterize(load_annotation(split, name), img.shape)
            cid = case_id(split, name)
            box = boxes[cid] = heart_roi_box(split, name)
            cv2.imwrite(str(out / f"images{g}" / f"{cid}_0000.png"),
                        crop(img, box, ROI_SIZE, cv2.INTER_LINEAR))
            cv2.imwrite(str(out / f"labels{g}" / f"{cid}.png"),
                        crop(lab, box, ROI_SIZE, cv2.INTER_NEAREST))
    (out / "crops.json").write_text(json.dumps(boxes, indent=1))

    generate_dataset_json(
        str(out), channel_names={0: "ultrasound"}, labels=LABELS,
        num_training_cases=len(groups["Tr"]), file_ending=".png", dataset_name=DATASET_NAME,
        description="Fetal four-chamber view: LV, RV, LA, RA. Images from FOCUS "
                    "(Zenodo 14597550) and FETAL_PLANES_DB (Zenodo 3904280), chamber "
                    "annotations made for this project.",
        reference="FOCUS + FETAL_PLANES_DB + own annotations", license="research / coursework")

    split = {"train": [case_id(s, n) for s, n in groups["Tr"] if role(s) == "train"],
             "val": [case_id(s, n) for s, n in groups["Tr"] if role(s) == "val"]}
    (out / "splits_final.json").write_text(json.dumps([split], indent=1))
    print(f"exported to {out}")
    print(f"  training {len(split['train'])}, validation {len(split['val'])}, "
          f"test {len(groups['Ts'])} images; labels {', '.join(CHAMBERS)}")
    return out


if __name__ == "__main__":
    export()
