"""Heart and thorax outline of unlabelled images with the stage-1 U-Net, plus quality checks.

Used for the external datasets (FETAL_PLANES_DB, Heartbeat), which have no masks.
They come from other machines and are not all clean four-chamber views, so every
prediction gets quality measures and a verdict:
    conf_*     mean U-Net probability inside the predicted region (how sure it is)
    fit_*      overlap of the raw region with its fitted ellipse (is it ellipse-shaped)
    inside     share of the heart ellipse inside the thorax ellipse
    thorax_cut share of the thorax ellipse outside the image (zoomed-in images)
    ctr        heart area / thorax area (FOCUS: 0.2 - 0.45)
"""
import cv2
import numpy as np
import torch

from src.data import STRUCTURES, ellipse_to_mask
from src.measurements import ellipse_area
from src.postprocess import fill_holes, fit_ellipse, largest_component
from src.preprocessing import prepare, restore_size

LIMITS = {"conf": 0.75, "fit": 0.85, "inside": 0.95, "thorax_cut": 0.10, "ctr": (0.15, 0.55)}
DOPPLER_FRACTION = 0.01  # more coloured pixels than this: colour Doppler (the U-Nets saw only B-mode)


def colour_fraction(bgr):
    """Share of clearly coloured pixels (colour Doppler); B-mode images are grey."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return float(((np.abs(b - g) + np.abs(g - r)) > 40).mean())


@torch.no_grad()
def predict_probs(model, cfg, img, device="cpu"):
    """(2, H, W) heart and thorax probabilities at the original image size."""
    x, meta = prepare(img, cfg=cfg, return_meta=True)
    p = torch.sigmoid(model(x[None].to(device)))[0].cpu().numpy()
    return np.stack([restore_size(p[c], meta["resize"], interp=cv2.INTER_LINEAR)
                     for c in range(len(STRUCTURES))])


def iou(a, b):
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def analyse(prob, shape):
    """prob: (2, H, W) probabilities -> ellipse masks (0/1) and the quality measures."""
    row, masks, ellipses = {}, {}, {}
    for c, k in enumerate(STRUCTURES):
        raw = (prob[c] > 0.5).astype(np.uint8)
        clean = fill_holes(largest_component(raw))
        e = fit_ellipse(clean) if clean.any() else None
        tag = "heart" if k == "cardiac" else "thorax"
        if e is None:
            row.update({f"conf_{tag}": 0.0, f"fit_{tag}": 0.0})
            masks[k], ellipses[k] = np.zeros(shape, np.uint8), None
            continue
        m = ellipse_to_mask(shape, e)
        row[f"conf_{tag}"] = float(prob[c][clean > 0].mean())
        row[f"fit_{tag}"] = iou(clean, m)
        masks[k], ellipses[k] = m, e
    h, t = ellipses["cardiac"], ellipses["thorax"]
    if h is None or t is None:
        row.update({"inside": 0.0, "thorax_cut": 1.0, "ctr": np.nan})
    else:
        hm, tm = masks["cardiac"] > 0, masks["thorax"] > 0
        row["inside"] = float((hm & tm).sum() / max(hm.sum(), 1))
        row["thorax_cut"] = float(max(0.0, 1 - tm.sum() / ellipse_area(t)))
        row["ctr"] = float(ellipse_area(h) / ellipse_area(t))
    return masks, row


def heart_found(r):
    """The heart outline alone is reliable (enough to crop the chambers)."""
    return r["conf_heart"] >= LIMITS["conf"] and r["fit_heart"] >= LIMITS["fit"]


def verdict(r):
    """Reasons ('a;b') why heart + thorax are not usable together (CTR); '' = usable."""
    reasons = []
    if min(r["conf_heart"], r["conf_thorax"]) < LIMITS["conf"]:
        reasons.append("unsure")
    if min(r["fit_heart"], r["fit_thorax"]) < LIMITS["fit"]:
        reasons.append("not elliptic")
    if r["inside"] < LIMITS["inside"]:
        reasons.append("heart outside thorax")
    if r["thorax_cut"] > LIMITS["thorax_cut"]:
        reasons.append("thorax cut off")
    lo, hi = LIMITS["ctr"]
    if not (lo <= r["ctr"] <= hi):
        reasons.append("implausible CTR")
    return ";".join(reasons)


def draw(img, masks, size=260):
    """Small RGB preview with the heart (red) and thorax (green) outlines."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, col in (("thorax", (60, 200, 60)), ("cardiac", (40, 40, 230))):
        cnt, _ = cv2.findContours(masks[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnt, -1, col, max(2, img.shape[1] // 300))
    s = size / max(img.shape)
    return cv2.cvtColor(cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
