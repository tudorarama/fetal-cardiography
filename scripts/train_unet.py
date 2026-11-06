"""Trains the U-Net on FOCUS (train split), selecting the epoch on the validation split.

Run from the project root:
    python scripts/train_unet.py                       # with preprocessing  -> results/unet_pre.pth
    python scripts/train_unet.py --no-preprocess       # ablation            -> results/unet_raw.pth
    python scripts/train_unet.py --epochs 2            # quick smoke test
    python scripts/train_unet.py --resume              # continue after an interruption

On CPU one epoch takes roughly 10-20 s; on a GPU (e.g. Google Colab) it is much faster.
The training curve is saved next to the checkpoint (results/unet_<name>_log.csv).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceLoss
from monai.utils import set_determinism

from src.unet import (FocusDataset, build_model, checkpoint_path, load_split_tensors,
                      train_transforms)


def dice_per_channel(logits, target, eps=1e-7):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    return (2 * inter / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--heart-size-aug", action="store_true",
                    help="synthetic heart enlargement/shrinking as augmentation (name suffix _cm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from results/unet_<name>_last.pth after an interruption")
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_pre = not args.no_preprocess
    name = ("pre" if use_pre else "raw") + ("_cm" if args.heart_size_aug else "")
    print(f"device {device}, preprocessing {use_pre}, heart size aug {args.heart_size_aug}, "
          f"epochs {args.epochs}")

    train_items = load_split_tensors("train", use_pre)
    val_items = load_split_tensors("val", use_pre)
    train_dl = torch.utils.data.DataLoader(FocusDataset(train_items,
                                                        train_transforms(args.heart_size_aug)),
                                           batch_size=args.batch, shuffle=True)
    val_dl = torch.utils.data.DataLoader(FocusDataset(val_items), batch_size=args.batch)

    model = build_model().to(device)
    dice_loss = DiceLoss(sigmoid=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, log, path = -1.0, [], checkpoint_path(name)
    last_path = path.with_name(f"unet_{name}_last.pth")
    path.parent.mkdir(exist_ok=True)
    start = 1
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"])
        best, log, start = state["best"], state["log"], state["epoch"] + 1
        print(f"resumed after epoch {state['epoch']} (best val Dice so far {best:.3f})")
    elif args.resume:
        print("no previous run to resume, starting from scratch")

    for epoch in range(start, args.epochs + 1):
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
                    "val_dice_cardiac": float(d[0]), "val_dice_thorax": float(d[1]),
                    "lr": sched.get_last_lr()[0]})
        mark = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": score,
                        "use_preprocess": use_pre}, path)
            mark = "  * saved"
        # full training state after every epoch, so an interrupted run can be resumed
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": torch.get_rng_state(),
                    "epoch": epoch, "best": best, "log": log}, last_path)
        pd.DataFrame(log).to_csv(path.with_name(f"unet_{name}_log.csv"), index=False)
        print(f"epoch {epoch:3d}  loss {log[-1]['train_loss']:.4f}  val Dice heart {d[0]:.3f} "
              f"thorax {d[1]:.3f}  ({time.time() - t0:.0f}s){mark}", flush=True)

    pd.DataFrame(log).to_csv(path.with_name(f"unet_{name}_log.csv"), index=False)
    print(f"best val Dice (mean of both) {best:.3f}, checkpoint {path}")


if __name__ == "__main__":
    main()
