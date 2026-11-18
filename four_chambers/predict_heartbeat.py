"""Segments the chambers of the Heartbeat four-chamber images and measures them.

Run from the four_chambers folder, after ../scripts/segment_heartbeat.py
(GPU, about 10 minutes for ~2,800 images):
    python predict_heartbeat.py

Step 2 of the anomaly test on Heartbeat (restricted access, CHD / No_CHD per patient):
    1. ../scripts/segment_heartbeat.py   heart outline (crop) and CTR, stage-1 U-Nets
    2. here                              chambers (nnU-Net v2) and their measurements
    3. ../scripts/evaluate_heartbeat.py  anomaly rules against the CHD label

Only images where the heart was found clearly (heart_ok) are segmented; the crop
is the same square around the U-Net heart outline as for FETAL_PLANES_DB. The
cardiac axis needs a spine landmark, which Heartbeat does not have, so only the
area measurements are computed:
    lv_rv, la_ra      area ratios
    frac_<chamber>    share of the four chambers' area (missing chamber < 3 %)
    coverage          chambers' area / heart outline area: low = the network found
                      little that looks like a four-chamber view (bad image, not CHD)
    solidity, layout_ok   measurements.segmentation_quality(): is the segmentation
                      plausible at all (one blob per chamber, ventricles next to their atria)
    flags             measurements.screen()
Outputs (local only):
    nnunet/predictions/heartbeat/<id>.png      label maps (0 background, 1 LV, 2 RV, 3 LA, 4 RA)
    ../results/heartbeat_chambers.csv          one row per image
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse

import cv2
import pandas as pd
from tqdm import tqdm

from common import roi_box
from config import CHAMBERS, PREDICTIONS_DIR, PROJECT, setup_nnunet_env
from measurements import chamber_measurements, screen, segmentation_quality
from predict import default_trainer, predict_rois

OUTLINE_CSV = PROJECT / "results" / "heartbeat_outline.csv"
OUT_CSV = PROJECT / "results" / "heartbeat_chambers.csv"
MASK_DIR = PROJECT / "data" / "HEARTBEAT_masks"
HEARTBEAT_DIR = PROJECT / "data" / "HEARTBEAT"


def image_path(row):
    """data/HEARTBEAT/<trimester>/<part>/<class>/<patient>/<image>.png from an outline row."""
    stem = row["id"].split("_", 1)[1]
    cls = "CHD" if row["chd"] else "No_CHD"
    return HEARTBEAT_DIR / row["trimester"] / row["part"] / cls / row["patient"] / f"{stem}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", help="default: the one used by the last train.py run")
    ap.add_argument("--limit", type=int, help="only the first N images (quick test)")
    ap.add_argument("--measure-only", action="store_true",
                    help="reuse the label maps of a previous run, only recompute the measurements")
    args = ap.parse_args()
    setup_nnunet_env()
    if not OUTLINE_CSV.exists():
        raise SystemExit(f"{OUTLINE_CSV} missing: run ../scripts/segment_heartbeat.py first")

    outline = pd.read_csv(OUTLINE_CSV)
    outline = outline[outline["heart_ok"]]
    if args.limit:
        outline = outline.head(args.limit)
    items, heart_area = [], {}
    for _, row in tqdm(outline.iterrows(), total=len(outline), desc="crops"):
        cid = row["id"]
        img = cv2.imread(str(image_path(row)), cv2.IMREAD_GRAYSCALE)
        heart = cv2.imread(str(MASK_DIR / f"{cid}-cardiac.png"), cv2.IMREAD_GRAYSCALE) > 0
        heart_area[cid] = float(heart.sum())
        items.append((cid, img, roi_box(heart)))
    if args.measure_only:
        pred_dir = PREDICTIONS_DIR / "heartbeat"
    else:
        pred_dir = predict_rois(items, "heartbeat", default_trainer(args.trainer))

    rows = []
    for cid in tqdm(heart_area, desc="measurements"):
        lab = cv2.imread(str(pred_dir / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
        m = chamber_measurements(lab)
        rows.append({"id": cid, "lv_rv": m["lv_rv"], "la_ra": m["la_ra"],
                     **{f"frac_{c}": m["fractions"][c] for c in CHAMBERS},
                     "coverage": sum(m["areas"].values()) / heart_area[cid],
                     **segmentation_quality(lab),
                     "flags": ";".join(screen(m))})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV} ({len(rows)} images)")


if __name__ == "__main__":
    main()
