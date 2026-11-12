"""Preprocess train, val, and test splits and cache tensors, masks, and metadata."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import PREPROC, UNET_SIZE
from src.data import list_cases, load_case
from src.preprocessing import crop_heart_roi, mask_to_unet, prepare


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def process_split(split, out_root, cfg):
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    names = list_cases(split)

    for name in names:
        img, _, masks = load_case(split, name)
        heart_mask = masks.get("cardiac") if masks else None

        tensor, meta = prepare(img, heart_mask=heart_mask, cfg=cfg, return_meta=True)
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        np.save(split_dir / f"{name}.npy", arr.astype(np.float32))

        if masks:
            proc_masks = {}
            for m_name, m_arr in masks.items():
                if cfg.get("crop", False) and heart_mask is not None:
                    m_crop, _ = crop_heart_roi(
                        m_arr, heart_mask, margin=cfg.get("roi_margin", 0.25)
                    )
                else:
                    m_crop = m_arr
                proc_masks[m_name] = mask_to_unet(m_crop, size=UNET_SIZE)
            np.savez_compressed(split_dir / f"{name}_masks.npz", **proc_masks)

        record = {
            "case": name,
            "split": split,
            "meta": _to_jsonable(meta),
            "config": _to_jsonable(cfg),
        }
        with open(split_dir / f"{name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    print(f"{split}: saved {len(names)} cases -> {split_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Cache preprocessed dataset splits.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "processed")
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.out_dir, PREPROC)


if __name__ == "__main__":
    main()