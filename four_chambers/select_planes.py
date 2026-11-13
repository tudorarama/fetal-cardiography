"""Chooses the FETAL_PLANES_DB images added to the four-chamber dataset.

Run from the four_chambers folder, after ../scripts/segment_planes.py (stage-1 U-Net
heart and thorax masks for every FETAL_PLANES_DB image):
    python select_planes.py                        # 150 train, 30 val, 30 test candidates
    python select_planes.py --train 200 --val 40 --test 40
    python select_planes.py --train all --val all --test 50   # every training patient

Run again with larger numbers to ADD images: the images already chosen stay
(their annotations keep their set), only new patients are appended.

The images are split BY PATIENT (several images of one fetus never end up in two
sets): the official FETAL_PLANES_DB test patients give the test candidates, its
training patients are divided into train and validation. One image per patient
(more variety), only images where the U-Net found a clear, elliptic heart.

They are candidates: FETAL_PLANES_DB "thorax" images are not all four-chamber
views, so in annotate.py the ones that are not are marked with x (not usable).
Writes planes_split.json: {"ptrain": [...], "pval": [...], "ptest": [...]}.
The split names ptrain / pval / ptest keep the two sources apart everywhere
(annotation files, nnU-Net case ids, evaluation per source).
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import PLANES_SPLIT_FILE, PROJECT

MASKS_CSV = PROJECT / "results" / "planes_masks.csv"
MIN_HEART_FIT = 0.85   # overlap of the U-Net heart region with its ellipse


def main():
    ap = argparse.ArgumentParser()
    count = lambda s: 10 ** 9 if s == "all" else int(s)  # noqa: E731
    ap.add_argument("--train", type=count, default=150, help="total number (or 'all')")
    ap.add_argument("--val", type=count, default=30, help="total number (or 'all')")
    ap.add_argument("--test", type=count, default=30, help="total number (or 'all')")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not MASKS_CSV.exists():
        raise SystemExit(f"missing {MASKS_CSV}: run ../scripts/segment_planes.py first")
    old = json.loads(PLANES_SPLIT_FILE.read_text()) if PLANES_SPLIT_FILE.exists() else {}

    df = pd.read_csv(MASKS_CSV)
    df = df[(df.fit_heart >= MIN_HEART_FIT) & (df.conf_thorax > 0)]
    rng = np.random.default_rng(args.seed)
    one = df.groupby("patient").sample(1, random_state=args.seed)  # one image per patient

    test_pat = one[one.official_train == 0]
    train_pat = one[one.official_train == 1].sample(frac=1, random_state=args.seed)
    n_val_pat = round(len(train_pat) * 0.2)
    val_pat, tr_pat = train_pat.iloc[:n_val_pat], train_pat.iloc[n_val_pat:]

    def take(rows, n):
        idx = rng.permutation(len(rows))[:n]
        return [str(x) for x in rows.iloc[idx]["name"]]

    split = {}
    for key, rows, n in (("ptrain", tr_pat, args.train), ("pval", val_pat, args.val),
                         ("ptest", test_pat, args.test)):
        have = list(old.get(key, []))
        assert set(have) <= set(rows["name"]), f"{key}: planes_split.json does not match the patient split"
        rest = rows[~rows["name"].isin(have)]
        split[key] = have + take(rest, max(0, n - len(have)))  # new images go to the end
    pats = {k: set(df.set_index("name").loc[v, "patient"]) for k, v in split.items()}
    assert not (pats["ptrain"] & pats["pval"] or pats["ptrain"] & pats["ptest"]
                or pats["pval"] & pats["ptest"]), "patient in two sets"
    PLANES_SPLIT_FILE.write_text(json.dumps(split, indent=1))
    print(f"saved {PLANES_SPLIT_FILE}: " + ", ".join(
        f"{k} {len(v)} (+{len(v) - len(old.get(k, []))})" for k, v in split.items()))
    print("next: python annotate.py --add-planes   (appends them to the annotation list)")


if __name__ == "__main__":
    main()
