"""Chooses the Heartbeat images to annotate for the four-chamber model.

Run from the four_chambers folder, after ../scripts/segment_heartbeat.py (it gives
the heart outline of every Heartbeat 4C image, used for the crop):
    python select_heartbeat.py            # writes annotations_heartbeat/heartbeat_split.json
    python select_heartbeat.py --force    # choose again (the annotations already drawn stay)

Why Heartbeat: FOCUS comes from one machine and nnU-Net segments only about a third
of the Heartbeat images plausibly. Annotating Heartbeat images teaches it the
other machines and gives a Dice on them, including hearts with a defect (CHD).

How (one image per patient, every patient in one set only):
    htrain / hval  No_CHD patients of the cross_val part, half 2T and half 3T.
                   Healthy hearts only, so that the model never learns what a CHD
                   heart looks like: the CHD test stays fair. These patients are
                   left out of the CHD evaluation (scripts/evaluate_heartbeat.py).
                   Annotated from the model's pre-drawing (predict.py --prefill).
    htest          half CHD (all CHD patients of the 2T test part first), half No_CHD,
                   half 2T and half 3T. Drawn by hand, never pre-filled.
Only images where the stage-1 U-Net found the heart clearly (heart_ok) are
candidates: the chamber model is applied only to those. Images that turn out not
to be a four-chamber view are marked with x in the annotation tool.

Heartbeat may not be redistributed: the list and the annotations stay in
annotations_heartbeat/ (git-ignored).
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse
import json

import numpy as np
import pandas as pd

from common import heartbeat_rel_path
from config import HEARTBEAT_ANN_DIR, HEARTBEAT_SPLIT_FILE, PROJECT

OUTLINE_CSV = PROJECT / "results" / "heartbeat_outline.csv"


def pick(candidates, n, rng, taken, prefer=None):
    """n patients not in taken (those of `prefer` first), one random image each."""
    pats = candidates.loc[~candidates["patient"].isin(taken), "patient"].unique()
    pats = rng.permutation(pats)
    if prefer is not None:
        first = set(candidates.loc[prefer, "patient"])
        pats = sorted(pats, key=lambda p: p not in first)  # stable: keeps the random order inside
    chosen = []
    for p in pats[:n]:
        rows = candidates[candidates["patient"] == p]
        chosen.append(rows.iloc[rng.integers(len(rows))])
        taken.add(p)
    if len(chosen) < n:
        print(f"  only {len(chosen)} of {n} patients available")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=90)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test", type=int, default=60, help="half CHD, half No_CHD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="replace an existing list")
    args = ap.parse_args()
    if HEARTBEAT_SPLIT_FILE.exists() and not args.force:
        raise SystemExit(f"{HEARTBEAT_SPLIT_FILE} exists (annotations may depend on it): use --force")
    if not OUTLINE_CSV.exists():
        raise SystemExit(f"{OUTLINE_CSV} missing: run ../scripts/segment_heartbeat.py first")

    df = pd.read_csv(OUTLINE_CSV)
    df = df[df["heart_ok"] & ~df["doppler"]]
    rng = np.random.default_rng(args.seed)
    taken = set()
    split = {"htrain": [], "hval": [], "htest": []}

    # test first, so that the few CHD patients of the official 2T test part all go there
    half = args.test // 2
    for chd in (1, 0):
        for tri, n in (("2T", half - half // 2), ("3T", half // 2)):
            c = df[(df["chd"] == chd) & (df["trimester"] == tri)]
            split["htest"] += pick(c, n, rng, taken, prefer=c["part"] == "test")

    healthy = df[(df["chd"] == 0) & (df["part"] == "cross_val")]
    for key, n in (("hval", args.val), ("htrain", args.train)):
        for tri, m in (("2T", n - n // 2), ("3T", n // 2)):
            split[key] += pick(healthy[healthy["trimester"] == tri], m, rng, taken)

    out = {k: [r["id"] for r in rows] for k, rows in split.items()}
    rows = [r for v in split.values() for r in v]
    out["paths"] = {r["id"]: heartbeat_rel_path(r) for r in rows}
    out["chd"] = {r["id"]: int(r["chd"]) for r in rows}
    out["patients"] = {k: sorted({r["patient"] for r in v}) for k, v in split.items()}
    pats = [set(out["patients"][k]) for k in ("htrain", "hval", "htest")]
    assert not (pats[0] & pats[1] or pats[0] & pats[2] or pats[1] & pats[2]), "patient in two sets"

    HEARTBEAT_ANN_DIR.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_SPLIT_FILE.write_text(json.dumps(out, indent=1))
    for k, v in split.items():
        t = pd.DataFrame(v)
        print(f"{k:6s} {len(v):3d} images: CHD {int(t['chd'].sum())}, "
              + ", ".join(f"{tri} {n}" for tri, n in t["trimester"].value_counts().sort_index().items())
              + f", official test part {int((t['part'] == 'test').sum())}")
    print(f"saved {HEARTBEAT_SPLIT_FILE}")


if __name__ == "__main__":
    main()
