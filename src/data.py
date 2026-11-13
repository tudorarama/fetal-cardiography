"""Loading the FOCUS and FETAL_PLANES_DB datasets.

FOCUS layout (data/FOCUS/):
    training/ (200)   validation/ (50)   testing/ (50)
        images/              001.png                 grayscale ultrasound image
        annfiles_ellipse/    001.txt                 one line per structure:
                                                     "cx cy a b angle label"
                                                     a, b = SEMI-axes (px), angle in degrees
        annfiles_mask/       001-cardiac.png         binary masks (0 / 255)
                             001-thorax.png
        annfiles_rectangle/  001.txt                 rotated boxes (not used)

FETAL_PLANES_DB layout (data/ZENODO/):
    Images/*.png
    FETAL_PLANES_DB_data.csv   ';'-separated, column "Train" = official split (1 = train)
    The project uses 485 "Fetal thorax" images (heart plane), one per patient:
    348 train, 87 validation, 50 test (four_chambers/planes_split.json). They
    were chosen from the 1,718 thorax images of the full dataset (12,400 images,
    Zenodo 3904280): patients have several near-identical images, and images
    where the heart was not found well were left out.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FOCUS_DIR = ROOT / "data" / "FOCUS"
PLANES_DIR = ROOT / "data" / "ZENODO"

SPLITS = {"train": "training", "val": "validation", "test": "testing"}
STRUCTURES = ("cardiac", "thorax")


# --------------------------------------------------------------------------- FOCUS

def _split_dir(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    return FOCUS_DIR / SPLITS[split]


def list_cases(split):
    """Case names (e.g. '001') in an official FOCUS split: 'train', 'val' or 'test'."""
    return sorted(p.stem for p in (_split_dir(split) / "images").glob("*.png"))


def load_image(path):
    """Grayscale uint8 image of shape (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_ellipses(split, name):
    """{'cardiac': (cx, cy, a, b, angle_deg), 'thorax': (...)} with a, b semi-axes in pixels."""
    path = _split_dir(split) / "annfiles_ellipse" / f"{name}.txt"
    ellipses = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            ellipses[parts[5]] = tuple(float(v) for v in parts[:5])
    missing = set(STRUCTURES) - set(ellipses)
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return ellipses


def load_masks(split, name):
    """{'cardiac': mask, 'thorax': mask}, each uint8 (H, W) with values 0/1."""
    d = _split_dir(split) / "annfiles_mask"
    return {k: (load_image(d / f"{name}-{k}.png") > 0).astype(np.uint8) for k in STRUCTURES}


def load_case(split, name):
    """Image, ground-truth ellipses and ground-truth masks of one FOCUS case."""
    img = load_image(_split_dir(split) / "images" / f"{name}.png")
    return img, load_ellipses(split, name), load_masks(split, name)


def ellipse_to_mask(shape, ellipse):
    """Filled binary mask (0/1) of an ellipse (cx, cy, a, b, angle_deg), a, b semi-axes."""
    cx, cy, a, b, angle = ellipse
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (round(cx), round(cy)), (round(a), round(b)),
                angle, 0, 360, color=1, thickness=-1)
    return mask


def overlay(img, masks, colors=None, thickness=2):
    """RGB image with the contour of each mask drawn on top.

    masks: dict name -> mask. Default colors: thorax green, cardiac red, others blue.
    """
    default = {"thorax": (0, 255, 0), "cardiac": (255, 0, 0)}
    colors = colors or {}
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    for name, m in masks.items():
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        color = colors.get(name, default.get(name, (0, 128, 255)))
        cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


# --------------------------------------------------------------- FETAL_PLANES_DB

def load_planes_table():
    """FETAL_PLANES_DB metadata as a DataFrame with stripped strings and an 'image_path' column."""
    import pandas as pd

    df = pd.read_csv(PLANES_DIR / "FETAL_PLANES_DB_data.csv", sep=";")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()
    df["Train"] = df["Train"].astype(int)
    df["image_path"] = [PLANES_DIR / "Images" / f"{n}.png" for n in df["Image_name"]]
    return df
