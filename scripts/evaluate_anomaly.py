"""Evaluates the cardiomegaly rule (CTR > 0.35) on real and synthetic images.

Run from the project root (after train_unet.py):
    python scripts/evaluate_anomaly.py                # U-Nets
    python scripts/evaluate_anomaly.py --classical    # also the classical method (slower)

1. Reference: normal CTR range from the ground truth of the TRAINING split
   -> results/anomaly_reference.json
2. Real test images: FOCUS has no enlarged heart in the test split, so this only
   measures false alarms (specificity).
3. Stress test: every test image is warped with src.synthetic.enlarge_heart for
   s in SCALES (heart grown by s, CTR by about s^2). The ground-truth CTR of the
   warped image is known, so sensitivity, specificity and AUC of the rule can be
   measured, as well as how well the predicted CTR follows the true one.
Outputs: results/anomaly_stress_test.csv, anomaly_stress_ctr.png, anomaly_stress_examples.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
from tqdm import tqdm

from src.anomaly import CTR_LIMIT, build_reference, load_reference, save_reference, screen
from src.data import list_cases, load_case, load_ellipses, overlay
from src.measurements import ctr_area
from src.postprocess import refine
from src.synthetic import enlarge_heart
from src.unet import make_predictor as unet_predictor

OUT = ROOT / "results"
SCALES = (1.0, 1.1, 1.2, 1.3, 1.4)


def auc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney U statistic."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        return np.nan
    return mannwhitneyu(scores[labels], scores[~labels]).statistic / (labels.sum() * (~labels).sum())


def predictors(with_classical):
    preds = {}
    for ckpt in sorted(OUT.glob("unet_*.pth")):
        if not ckpt.stem.endswith("_last"):
            preds[ckpt.stem] = unet_predictor(ckpt)
    if with_classical:
        from src.classical import load_params, make_predictor
        preds["classical"] = make_predictor(load_params(OUT / "classical_params.json"))
    return preds


def stress_test(preds, reference, split="test"):
    rows = []
    for name in tqdm(list_cases(split), desc="stress test"):
        img, _, gt = load_case(split, name)
        for s in SCALES:
            w, wm = enlarge_heart(img, gt, s)
            gt_ell = {k: refine(wm[k])[1] for k in wm}
            ctr_gt = ctr_area(gt_ell["cardiac"], gt_ell["thorax"])
            for method, predict in preds.items():
                _, ell = predict(w)
                ok = ell.get("cardiac") is not None and ell.get("thorax") is not None
                ctr = ctr_area(ell["cardiac"], ell["thorax"]) if ok else np.nan
                codes = {f.code for f in screen({"ctr_area": ctr}, reference)} if ok else set()
                rows.append({"name": name, "scale": s, "method": method, "ctr_gt": ctr_gt,
                             "ctr_pred": ctr, "flag": "cardiomegaly" in codes,
                             "flag_z": "ctr_outside_reference" in codes})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for method, d in df.groupby("method"):
        pos = d["ctr_gt"] > CTR_LIMIT
        r = {"method": method, "positives": int(pos.sum()), "negatives": int((~pos).sum()),
             "sensitivity": d.loc[pos, "flag"].mean(), "specificity": 1 - d.loc[~pos, "flag"].mean(),
             "auc": auc(d["ctr_pred"].fillna(0), pos),
             "pearson_r": pearsonr(d["ctr_gt"], d["ctr_pred"].fillna(0))[0],
             "ctr_abs_error": (d["ctr_pred"] - d["ctr_gt"]).abs().mean()}
        for s, ds in d.groupby("scale"):
            r[f"ctr_pred@{s}"] = ds["ctr_pred"].mean()
            r[f"ctr_gt@{s}"] = ds["ctr_gt"].mean()
        out.append(r)
    return pd.DataFrame(out).set_index("method").T


def ctr_figure(df, path):
    methods = list(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5), squeeze=False)
    lim = (0.1, max(df["ctr_gt"].max(), df["ctr_pred"].max()) + 0.03)
    for ax, m in zip(axes[0], methods):
        d = df[df["method"] == m]
        sc = ax.scatter(d["ctr_gt"], d["ctr_pred"], c=d["scale"], cmap="viridis", s=14)
        ax.plot(lim, lim, "k--", lw=1)
        ax.axvline(CTR_LIMIT, color="r", lw=1)
        ax.axhline(CTR_LIMIT, color="r", lw=1)
        ax.set(xlim=lim, ylim=lim, xlabel="true CTR (warped ground truth)",
               ylabel="predicted CTR", title=m)
        ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[0].tolist(), label="enlargement factor s")
    fig.suptitle(f"Synthetic cardiomegaly: red lines = threshold {CTR_LIMIT}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def examples_figure(predict, method, path, name="001", split="test"):
    img, _, gt = load_case(split, name)
    fig, axes = plt.subplots(1, len(SCALES), figsize=(4.2 * len(SCALES), 3.6))
    for ax, s in zip(axes, SCALES):
        w, wm = enlarge_heart(img, gt, s)
        masks, ell = predict(w)
        gt_ell = {k: refine(wm[k])[1] for k in wm}
        vis = overlay(overlay(w, wm, thickness=3), masks,
                      colors={"thorax": (255, 255, 0), "cardiac": (255, 0, 255)})
        ax.imshow(vis)
        ax.set_title(f"s={s}: CTR true {ctr_area(gt_ell['cardiac'], gt_ell['thorax']):.2f}, "
                     f"pred {ctr_area(ell['cardiac'], ell['thorax']):.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{method}, case {name}: GT green/red, prediction yellow/magenta")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", action="store_true", help="also run the classical method")
    args = ap.parse_args()

    # 1. normal range from training ground truth
    ctr_train = [ctr_area(e["cardiac"], e["thorax"])
                 for e in (load_ellipses("train", n) for n in list_cases("train"))]
    ref_path = OUT / "anomaly_reference.json"
    save_reference(build_reference(ctr_train), ref_path)
    reference = load_reference(ref_path)
    lo = reference["ctr_mean"] - 2.5 * reference["ctr_std"]
    hi = reference["ctr_mean"] + 2.5 * reference["ctr_std"]
    print(f"reference CTR (train GT): mean {reference['ctr_mean']:.3f}, "
          f"std {reference['ctr_std']:.3f} -> z-range [{lo:.3f}, {hi:.3f}]")

    # 2. real test images: false alarms only (no enlarged heart in FOCUS test)
    print("\nreal test images, flags (ground truth has none above the limit):")
    for csv in sorted(OUT.glob("*_test.csv")):
        if csv.stem.split("_test")[0] in ("classical", "prior") or (
                csv.stem.startswith("unet_") and "chambers" not in csv.stem):
            d = pd.read_csv(csv)
            n_gt = int((d["ctr_gt"] > CTR_LIMIT).sum())
            n_flag = int((d["ctr_pred"] > CTR_LIMIT).sum())
            print(f"  {csv.stem:18s} CTR > {CTR_LIMIT}: {n_flag:2d}/{len(d)}  (ground truth {n_gt})")

    # 3. stress test
    preds = predictors(args.classical)
    df = stress_test(preds, reference)
    df.to_csv(OUT / "anomaly_stress_test.csv", index=False)
    print("\n=== synthetic cardiomegaly, test split x scales", SCALES)
    print(summarize(df).round(3).to_string())
    ctr_figure(df, OUT / "anomaly_stress_ctr.png")
    first = next(iter(preds))
    examples_figure(preds[first], first, OUT / "anomaly_stress_examples.png")
    print(f"\nresults saved to {OUT}")


if __name__ == "__main__":
    main()
