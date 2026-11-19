"""Ablation study of the preprocessing: which filters help the segmentation, and by how much?

Each variant switches the steps of src.preprocessing.preprocess on or off:

    none               no filter (the raw image)
    median             median filter only
    bilateral          bilateral filter only
    median_bilateral   median + bilateral
    clahe              CLAHE only
    full               median + bilateral + CLAHE (the project's pipeline)

U-Net (heart + thorax): for every variant
    1. the z-score statistics are computed on the TRAIN images preprocessed with
       that variant (same computation as scripts/compute_norm_stats.py), so every
       network gets centred inputs;
    2. the U-Net is trained SEEDS times with different random seeds (same model,
       loss, optimiser and schedule as scripts/train_unet.py; the epoch is chosen
       on the validation split);
    3. every trained network is evaluated on the TEST split.
   A single training per variant cannot separate the effect of the filters from
   the luck of the random initialisation; the spread over seeds shows how large
   that luck is.

Classical method (Otsu + Chan-Vese): it has no training, so every variant is run
once on the TEST split with the parameters of models/classical_params.json (tuned
with the full pipeline and the stronger bilateral filter of src.classical).

Run from the project root:
    python scripts/ablation_preprocessing.py --smoke          # 2-3 min check that everything runs
    python scripts/ablation_preprocessing.py                  # everything (long on CPU, see below)
    python scripts/ablation_preprocessing.py --variants none full --seeds 3
    python scripts/ablation_preprocessing.py --skip-unet      # classical method only (~10 min)

Time: one U-Net training is about 100 epochs x 10-20 s on CPU (20-35 min), much less
on a GPU. All 6 variants x 3 seeds = 18 trainings. Finished runs are skipped when
the script is started again, so it can be stopped and resumed.

Outputs in results/ablation/ (not versioned):
    norm_stats.json                      mean / std of the network input, per variant
    unet_<variant>_s<seed>.pth / _log.csv / _test.csv
    classical_<variant>_test.csv
    summary_unet.csv, summary_classical.csv   mean +- std, paired tests against "none"
    ablation_preprocessing.png           bar charts for the report
    variants_demo.png                    the same image under every variant
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PREPROC, RESULTS_DIR
from src.data import list_cases, load_case
from src.preprocessing import preprocess, to_unet_input

# name -> (median kernel, bilateral filter, CLAHE clip limit); 0 / False = step switched off
VARIANTS = {
    "none": (0, False, 0.0),
    "median": (5, False, 0.0),
    "bilateral": (0, True, 0.0),
    "median_bilateral": (5, True, 0.0),
    "clahe": (0, False, 2.0),
    "full": (5, True, 2.0),
}
LABELS = {"none": "none", "median": "median", "bilateral": "bilateral",
          "median_bilateral": "median +\nbilateral", "clahe": "CLAHE", "full": "median +\nbilateral +\nCLAHE"}
METRICS = ("dice_cardiac", "dice_thorax", "hd95_cardiac", "hd95_thorax")


def variant_cfg(name, base=PREPROC):
    """Preprocessing settings of a variant, on top of base (PREPROC or the classical settings)."""
    k, bilateral, clip = VARIANTS[name]
    return {**base, "filter": True, "median": k, "bilateral": bilateral, "clahe": clip}


# ------------------------------------------------------------------ normalization

def norm_stats(cfg):
    """Mean and std (0-255 scale) of the network input on the TRAIN split, as in compute_norm_stats.py."""
    total = s = s2 = 0.0
    for name in list_cases("train"):
        img, _, _ = load_case("train", name)
        x, _ = to_unet_input(preprocess(img, cfg))   # letterboxed, [0, 1]
        x = x.astype(np.float64) * 255.0
        total += x.size
        s += x.sum()
        s2 += (x ** 2).sum()
    mean = s / total
    return float(mean), float(np.sqrt(s2 / total - mean ** 2))


def load_norm(out, variants):
    """{variant: (mean, std)}, computed once and kept in out/norm_stats.json."""
    path = out / "norm_stats.json"
    stats = json.loads(path.read_text()) if path.exists() else {}
    for v in variants:
        if v not in stats:
            print(f"normalization for '{v}' ...", flush=True)
            stats[v] = norm_stats(variant_cfg(v))
            path.write_text(json.dumps(stats, indent=2))
        print(f"  {v:17s} mean {stats[v][0]:6.2f}  std {stats[v][1]:6.2f}")
    return stats


# ------------------------------------------------------------------ U-Net

def dice_per_channel(logits, target, eps=1e-7):
    import torch
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    return (2 * inter / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)).mean(dim=0)


def train_one(train_items, val_items, cfg, seed, ckpt, log_path, epochs, batch, lr, device):
    """Same training as scripts/train_unet.py; saves the best epoch (validation Dice) to ckpt."""
    import torch
    from monai.losses import DiceLoss
    from monai.utils import set_determinism

    from src.unet import FocusDataset, build_model

    set_determinism(seed=seed)
    train_dl = torch.utils.data.DataLoader(FocusDataset(train_items, train=True, seed=seed),
                                           batch_size=batch, shuffle=True)
    val_dl = torch.utils.data.DataLoader(FocusDataset(val_items), batch_size=batch)
    model = build_model().to(device)
    dice_loss = DiceLoss(sigmoid=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, log = -1.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = dice_loss(out, y) + bce(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            d = torch.stack([dice_per_channel(model(x.to(device)), y.to(device))
                             for x, y in val_dl]).mean(dim=0).cpu().numpy()
        score = float(d.mean())
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                    "val_dice_cardiac": float(d[0]), "val_dice_thorax": float(d[1])})
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": score,
                        "preproc": cfg, "seed": seed}, ckpt)
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"    epoch {epoch:3d}  loss {log[-1]['train_loss']:.4f}  val Dice heart {d[0]:.3f} "
              f"thorax {d[1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return best


def run_unet(out, variants, seeds, stats, epochs, batch, lr, device, test_names):
    from src.evaluation import evaluate
    from src.unet import load_split, make_predictor

    for v in variants:
        cfg = {**variant_cfg(v), "mean": stats[v][0], "std": stats[v][1]}
        todo = [s for s in range(seeds) if not (out / f"unet_{v}_s{s}_test.csv").exists()]
        if not todo:
            print(f"U-Net '{v}': all {seeds} seeds done")
            continue
        print(f"U-Net '{v}': loading train/val with this preprocessing ...", flush=True)
        train_items, val_items = load_split("train", cfg), load_split("val", cfg)
        for s in todo:
            tag = f"unet_{v}_s{s}"
            print(f"  {tag}: training ({epochs} epochs)", flush=True)
            best = train_one(train_items, val_items, cfg, s, out / f"{tag}.pth",
                             out / f"{tag}_log.csv", epochs, batch, lr, device)
            print(f"  {tag}: best val Dice {best:.3f}; testing ...", flush=True)
            df = evaluate(make_predictor(out / f"{tag}.pth", device), "test", names=test_names)
            df["variant"], df["seed"] = v, s
            df.to_csv(out / f"{tag}_test.csv", index=False)
            print(f"  {tag}: test Dice heart {df.dice_cardiac.mean():.3f} thorax {df.dice_thorax.mean():.3f}"
                  f"  HD95 heart {df.hd95_cardiac.mean():.1f} px", flush=True)


# ------------------------------------------------------------------ classical method

def run_classical(out, variants, test_names):
    from src.classical import PREPROC_CLASSICAL, load_params, make_predictor
    from src.evaluation import evaluate

    params = load_params(MODELS_DIR / "classical_params.json")
    for v in variants:
        csv = out / f"classical_{v}_test.csv"
        if csv.exists():
            print(f"classical '{v}': done")
            continue
        print(f"classical '{v}' ...", flush=True)
        df = evaluate(make_predictor(params, variant_cfg(v, PREPROC_CLASSICAL)), "test",
                      names=test_names, progress=True)
        df["variant"] = v
        df.to_csv(csv, index=False)


# ------------------------------------------------------------------ summaries

def read(out, pattern):
    files = sorted(out.glob(pattern))
    return pd.concat([pd.read_csv(f, dtype={"name": str}) for f in files], ignore_index=True) if files else None


def paired_test(per_image, v, ref="none"):
    """Wilcoxon signed-rank p-values, variant v against ref, paired per test image."""
    from scipy.stats import wilcoxon
    out = {}
    if ref not in per_image or v == ref:
        return out
    a, b = per_image[v], per_image[ref]
    common = a.index.intersection(b.index)
    for m in METRICS:
        d = (a.loc[common, m] - b.loc[common, m]).dropna()
        try:
            out[f"p_{m}"] = float(wilcoxon(d).pvalue) if len(d) and d.abs().sum() > 0 else np.nan
        except ValueError:
            out[f"p_{m}"] = np.nan
    return out


def summarize_unet(out):
    df = read(out, "unet_*_s*_test.csv")
    if df is None:
        return None
    runs = df.groupby(["variant", "seed"])[list(METRICS)].mean()     # one row per training
    per_image = {v: g.groupby("name")[list(METRICS)].mean() for v, g in df.groupby("variant")}  # mean over seeds
    rows = []
    for v in [v for v in VARIANTS if v in runs.index.get_level_values(0)]:
        r = runs.loc[v]
        row = {"variant": v, "seeds": len(r)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = r[m].mean(), r[m].std(ddof=1) if len(r) > 1 else np.nan
        row.update(paired_test(per_image, v))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_unet.csv", index=False)
    return summary


def summarize_classical(out):
    df = read(out, "classical_*_test.csv")
    if df is None:
        return None
    per_image = {v: g.set_index("name")[list(METRICS)] for v, g in df.groupby("variant")}
    rows = []
    for v in [v for v in VARIANTS if v in per_image]:
        row = {"variant": v, **{f"{m}_mean": per_image[v][m].mean() for m in METRICS},
               "failed": int(df.loc[df.variant == v, "failed"].sum())}
        row.update(paired_test(per_image, v))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_classical.csv", index=False)
    return summary


def print_table(summary, title, with_std):
    print(f"\n=== {title} (test set; p = Wilcoxon against 'none', paired per image)")
    for _, r in summary.iterrows():
        cells = []
        for m, fmt in (("dice_cardiac", ".3f"), ("dice_thorax", ".3f"), ("hd95_cardiac", ".1f")):
            s = f"{r[f'{m}_mean']:{fmt}}"
            if with_std and not np.isnan(r[f"{m}_std"]):
                s += f" ± {r[f'{m}_std']:{fmt}}"
            p = r.get(f"p_{m}", np.nan)
            cells.append(s + ("" if pd.isna(p) else f" (p={p:.3f})"))
        print(f"{r['variant']:17s} Dice heart {cells[0]:24s} Dice thorax {cells[1]:24s} HD95 heart {cells[2]}")


def figure(out, su, sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if su is not None:
        panels += [(su, "dice_cardiac", "U-Net: heart Dice", True), (su, "hd95_cardiac", "U-Net: heart HD95 (px)", True)]
    if sc is not None:
        panels += [(sc, "dice_cardiac", "Classical: heart Dice", False)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8), squeeze=False)
    for ax, (s, m, title, std) in zip(axes[0], panels):
        x = np.arange(len(s))
        err = s[f"{m}_std"].fillna(0) if std and f"{m}_std" in s else None
        # points, not bars: the axis does not start at 0, and bars would exaggerate small gaps
        ax.errorbar(x, s[f"{m}_mean"], yerr=err, fmt="o", markersize=9, capsize=6, color="#2E5375")
        ax.set_xticks(x, [LABELS[v] for v in s["variant"]], fontsize=9)
        ax.set_title(title)
        lo, hi = s[f"{m}_mean"].min(), s[f"{m}_mean"].max()
        pad = max((hi - lo) * 0.8, 0.01 if "dice" in m else 0.5)
        ax.set_ylim(max(0, lo - pad), hi + pad)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Preprocessing ablation, FOCUS test set (U-Net: mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out / "ablation_preprocessing.png", dpi=150)
    plt.close(fig)


def variants_demo(out, split="train", name="001"):
    """The same image (around the heart) under every variant, for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, _, gt = load_case(split, name)
    ys, xs = np.nonzero(gt["thorax"])
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.2 * len(VARIANTS), 3.6))
    for ax, v in zip(axes, VARIANTS):
        ax.imshow(preprocess(img, variant_cfg(v))[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax.set_title(LABELS[v].replace("\n", " "), fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "variants_demo.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3, help="trainings per variant (seeds 0..N-1)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="quick check: variants none + full, 1 seed, 2 epochs, 4 test images, "
                         "written to results/ablation_smoke/")
    args = ap.parse_args()

    out = RESULTS_DIR / "ablation"
    test_names = None
    if args.smoke:
        out = RESULTS_DIR / "ablation_smoke"
        args.variants, args.seeds, args.epochs = ["none", "full"], 1, 2
        test_names = list_cases("test")[:4]
    out.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; variants {args.variants}; {args.seeds} seed(s) x {args.epochs} epochs")

    variants_demo(out)
    if not args.skip_unet:
        stats = load_norm(out, args.variants)
        run_unet(out, args.variants, args.seeds, stats, args.epochs, args.batch, args.lr, device, test_names)
    if not args.skip_classical:
        run_classical(out, args.variants, test_names)

    su, sc = summarize_unet(out), summarize_classical(out)
    if su is not None:
        print_table(su, "U-Net, mean ± std over seeds", with_std=True)
    if sc is not None:
        print_table(sc, "Classical method", with_std=False)
    figure(out, su, sc)
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()