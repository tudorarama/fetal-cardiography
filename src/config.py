"""Shared project constants: paths, preprocessing and augmentation settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # datasets (not versioned)
MODELS_DIR = ROOT / "models"      # trained U-Nets and tuned parameters (versioned)
RESULTS_DIR = ROOT / "results"    # tables and figures written by the scripts (not versioned)

SEED = 42

UNET_SIZE = 256

NORM_MEAN = 47.00
NORM_STD = 61.41

PREPROC = {
    "crop": False,
    "roi_margin": 0.25,
    "filter": True,
    "median": 5,
    "bilateral": True,
    "bilateral_d": 9,
    "bilateral_sigma_color": 50,
    "bilateral_sigma_space": 50,
    "clahe": 2.0,
    "clahe_tile": 8,
    "normalize": "zscore",
}

AUGMENT = {
    "p": 0.8,
    "max_angle": 15.0,
    "zoom_range": (0.9, 1.1),
    "brightness": 0.15,
    "flip": 0.5,      # left-right mirror (an upside-down sector never occurs)
    "shift": 0.08,    # random translation, as a fraction of the image side
}