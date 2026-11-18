"""Tests the anomaly rules on Heartbeat: do they flag the CHD patients more often?

Run from the project root, after scripts/segment_heartbeat.py and
four_chambers/predict_heartbeat.py:
    python scripts/evaluate_heartbeat.py

Heartbeat only says CHD / No_CHD per patient (no type of defect, no outlines).
The unit is an examination: one patient in one trimester (2T or 3T; some patients
were scanned in both). An exam has several 4C images; a rule flags the exam when
it flags the MAJORITY of its usable images, so one bad image does not decide.

Chamber rules only use images whose segmentation is plausible (one piece per
chamber, each ventricle next to its atrium): on these other machines nnU-Net
often fails, and a failed segmentation would look like an anomaly.

Rules (fixed in advance, textbook thresholds of src.anomaly):
    cardiomegaly             CTR > 0.35 (U-Net trained with random heart sizes)
    missing chamber          a chamber < 3 % of the four chambers' area
    ventricle disproportion  LV/RV outside 0.5-2
    atrium disproportion     LA/RA outside 0.5-2
    any rule                 at least one of the above
The cardiac axis is not tested: it needs a spine landmark, which Heartbeat lacks.

Reported:
    1. sensitivity (CHD exams flagged) and specificity (No_CHD exams not flagged)
       of every rule, on all exams and per trimester;
    2. AUC of the underlying measurement (exam median) - how well it separates
       CHD from No_CHD whatever the threshold;
    3. thresholds re-chosen on the cross_val exams (Youden index), checked on the
       separate 2T test exams (only 6 CHD: indicative).
Only aggregated numbers and plots are written (the data may not be redistributed):
    results/heartbeat_rules.csv, results/heartbeat_auc.csv, results/heartbeat_measurements.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.anomaly import CTR_LIMIT, MIN_CHAMBER_FRACTION, RATIO_RANGE
from src.config import RESULTS_DIR as OUT

CHAMBERS = ("LV", "RV", "LA", "RA")
MIN_COVERAGE = 0.25    # chambers cover < 25 % of the heart outline: not a usable 4C view
MIN_SOLIDITY = 0.9     # a chamber split into pieces: failed segmentation (FOCUS test: >= 0.99)

# measurement per image -> (exam score: higher = more abnormal, label)
SCORES = {
    "ctr": "CTR",
    "ventricle_dev": "|log LV/RV|",
    "atrium_dev": "|log LA/RA|",
    "min_fraction_neg": "- smallest chamber fraction",
}


def load(quality_filter=True):
    outline = pd.read_csv(OUT / "heartbeat_outline.csv")
    chambers_csv = OUT / "heartbeat_chambers.csv"
    if not chambers_csv.exists():
        raise SystemExit(f"{chambers_csv} missing: run four_chambers/predict_heartbeat.py first")
    df = outline.merge(pd.read_csv(chambers_csv), on="id", how="left")
    df["chambers_ok"] = df["heart_ok"] & (df["coverage"] >= MIN_COVERAGE)
    if quality_filter:  # plausible segmentation only (four_chambers/measurements.segmentation_quality)
        df["chambers_ok"] &= (df["solidity"] >= MIN_SOLIDITY) & (df["layout_ok"] != False)  # noqa: E712

    ctr = df["ctr_cm"].where(df["ctr_ok"])
    df["ctr"] = ctr
    df["r_cardiomegaly"] = (ctr > CTR_LIMIT).where(ctr.notna())
    ok = df["chambers_ok"]
    fr = df[[f"frac_{c}" for c in CHAMBERS]]
    df["min_fraction_neg"] = (-fr.min(axis=1)).where(ok)
    lo, hi = RATIO_RANGE
    df["r_missing_chamber"] = (fr.min(axis=1) < MIN_CHAMBER_FRACTION).where(ok)
    for ratio, dev, rule in (("lv_rv", "ventricle_dev", "r_ventricle_disproportion"),
                             ("la_ra", "atrium_dev", "r_atrium_disproportion")):
        v = df[ratio].astype(float)
        v = v.where(ok & (v > 0))  # a missing chamber is the other rule
        df[dev] = np.abs(np.log(v))
        df[rule] = (~v.between(lo, hi)).where(v.notna())
    return df


RULES = ("cardiomegaly", "missing_chamber", "ventricle_disproportion", "atrium_disproportion")


def per_exam(df):
    """One row per exam: majority vote of each rule over its usable images, median scores."""
    g = df.groupby("exam")
    ex = g[["chd", "trimester", "part"]].first()
    ex["images"] = g.size()
    for r in RULES:
        col = df[f"r_{r}"].astype(float)
        ex[f"n_{r}"] = col.notna().groupby(df["exam"]).sum()
        ex[r] = (col.groupby(df["exam"]).mean() > 0.5).where(ex[f"n_{r}"] > 0)
    usable = ex[list(RULES)].notna().any(axis=1)
    ex["any_rule"] = ex[list(RULES)].fillna(False).astype(bool).any(axis=1).where(usable)
    for s in SCORES:
        ex[s] = df.groupby("exam")[s].median()
    return ex


def sens_spec(flag, chd):
    m = flag.notna()
    f, y = flag[m].astype(bool), chd[m].astype(bool)
    return {"CHD exams": int(y.sum()), "No_CHD exams": int((~y).sum()),
            "sensitivity": f[y].mean() if y.any() else np.nan,
            "specificity": (~f[~y]).mean() if (~y).any() else np.nan}


def auc(score, chd):
    m = score.notna()
    pos, neg = score[m & (chd == 1)], score[m & (chd == 0)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, 0, 0
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return u / (len(pos) * len(neg)), p, len(pos), len(neg)


def youden_threshold(score, chd):
    m = score.notna()
    s, y = score[m].values, chd[m].values.astype(bool)
    best, thr = -1, np.nan
    for t in np.unique(s):
        j = (s[y] >= t).mean() + (s[~y] < t).mean() - 1
        if j > best:
            best, thr = j, t
    return thr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-quality-filter", action="store_true",
                    help="also use implausible chamber segmentations (to see their effect)")
    args = ap.parse_args()
    df = load(not args.no_quality_filter)
    ex = per_exam(df)
    n_img = len(df)
    print(f"{n_img} 4C images, {len(ex)} exams ({int(ex.chd.sum())} CHD)")
    print(f"usable images: chambers {df.chambers_ok.mean():.0%}, CTR {df.ctr_ok.mean():.0%} "
          f"(Doppler {df.doppler.mean():.0%})")
    print("usable share by class (a large gap would bias the comparison):")
    print(df.groupby("chd")[["chambers_ok", "ctr_ok"]].mean().round(2).to_string())

    # 1. fixed rules
    rows = []
    for subset, sub in [("all", ex), *[(f"{t}", g) for t, g in ex.groupby("trimester")]]:
        for r in (*RULES, "any_rule"):
            rows.append({"exams": subset, "rule": r, **sens_spec(sub[r], sub["chd"])})
    rules = pd.DataFrame(rows)
    rules.to_csv(OUT / "heartbeat_rules.csv", index=False)
    print("\n=== 1. rules with the fixed thresholds (exam = majority of its images)")
    print(rules.round(3).to_string(index=False))

    # 2. separation by the measurement itself
    rows = []
    for s, label in SCORES.items():
        a, p, npos, nneg = auc(ex[s], ex["chd"])
        rows.append({"measurement": label, "AUC": a, "p (Mann-Whitney)": p,
                     "CHD exams": npos, "No_CHD exams": nneg,
                     "median CHD": ex.loc[ex.chd == 1, s].median(),
                     "median No_CHD": ex.loc[ex.chd == 0, s].median()})
    aucs = pd.DataFrame(rows)
    aucs.to_csv(OUT / "heartbeat_auc.csv", index=False)
    print("\n=== 2. AUC of each measurement (exam median; 0.5 = no separation)")
    print(aucs.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # 3. thresholds chosen on cross_val, checked on the 2T test exams
    dev, test = ex[ex.part == "cross_val"], ex[ex.part == "test"]
    print(f"\n=== 3. thresholds re-chosen on cross_val ({int(dev.chd.sum())} CHD exams), "
          f"checked on 2T test ({int(test.chd.sum())} CHD exams, indicative)")
    flags = []
    for s, label in SCORES.items():
        t = youden_threshold(dev[s], dev["chd"])
        f_dev, f_test = (dev[s] >= t).where(dev[s].notna()), (test[s] >= t).where(test[s].notna())
        flags.append(f_test)
        d, e = sens_spec(f_dev, dev["chd"]), sens_spec(f_test, test["chd"])
        print(f"  {label:28s} >= {t:7.3f}   cross_val sens {d['sensitivity']:.2f} spec {d['specificity']:.2f}"
              f"   | test sens {e['sensitivity']:.2f} spec {e['specificity']:.2f}")

    # figure: exam medians, CHD vs No_CHD (aggregated, no images)
    fig, axes = plt.subplots(1, len(SCORES), figsize=(4 * len(SCORES), 4))
    for ax, (s, label) in zip(axes, SCORES.items()):
        data = [ex.loc[(ex.chd == k) & ex[s].notna(), s] for k in (0, 1)]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks([1, 2], [f"No_CHD\n(n={len(data[0])})", f"CHD\n(n={len(data[1])})"])
        a = aucs.set_index("measurement").loc[label, "AUC"]
        ax.set_title(f"{label}\nAUC {a:.2f}")
    fig.suptitle("Heartbeat 4C, per exam (median over its images)")
    fig.tight_layout()
    fig.savefig(OUT / "heartbeat_measurements.png", dpi=100)
    print(f"\nsaved heartbeat_rules.csv, heartbeat_auc.csv, heartbeat_measurements.png in {OUT}")


if __name__ == "__main__":
    main()
