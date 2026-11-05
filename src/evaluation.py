"""Evaluation shared by every segmentation method.

A method is any function  predict(img) -> (masks, ellipses)  where masks and
ellipses are dicts with keys 'cardiac' and 'thorax' (an ellipse may be None
when the method failed). The same per-case metrics and measurements are then
computed for the classical method and for U-Net.
"""
import numpy as np
import pandas as pd

from src.data import STRUCTURES, list_cases, load_case
from src.measurements import cardiac_axis, ctr_area
from src.metrics import dice, hd95


def evaluate_case(pred_masks, pred_ell, gt_masks, gt_ell):
    row = {}
    for k in STRUCTURES:
        row[f"dice_{k}"] = dice(pred_masks[k], gt_masks[k])
        row[f"hd95_{k}"] = hd95(pred_masks[k], gt_masks[k])
    ok = pred_ell.get("cardiac") is not None and pred_ell.get("thorax") is not None
    row["failed"] = not ok
    row["ctr_gt"] = ctr_area(gt_ell["cardiac"], gt_ell["thorax"])
    row["ctr_pred"] = ctr_area(pred_ell["cardiac"], pred_ell["thorax"]) if ok else np.nan
    row["axis_gt"] = cardiac_axis(gt_ell["cardiac"], gt_ell["thorax"])
    row["axis_pred"] = cardiac_axis(pred_ell["cardiac"], pred_ell["thorax"]) if ok else np.nan
    return row


def evaluate(predict, split, names=None, progress=False):
    """Runs predict on every case of a split and returns a per-case DataFrame."""
    names = names if names is not None else list_cases(split)
    if progress:
        from tqdm import tqdm
        names = tqdm(names, desc=split)
    rows = []
    for name in names:
        img, gt_ell, gt_masks = load_case(split, name)
        masks, ell = predict(img)
        rows.append({"split": split, "name": name,
                     **evaluate_case(masks, ell, gt_masks, gt_ell)})
    return pd.DataFrame(rows)


def summarize(df):
    """Mean +- std of the main metrics, as a one-row-per-metric DataFrame."""
    cols = [c for c in df.columns if c.startswith(("dice_", "hd95_"))]
    out = pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std(),
                        "median": df[cols].median()})
    err = (df["ctr_pred"] - df["ctr_gt"]).abs()
    out.loc["ctr_abs_error"] = [err.mean(), err.std(), err.median()]
    out.loc["failed_cases"] = [df["failed"].sum(), np.nan, np.nan]
    return out
