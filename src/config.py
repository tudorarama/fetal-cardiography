"""Shared project constants."""

SEED = 42

UNET_SIZE = 256

NORM_MEAN = 45.26
NORM_STD = 48.37

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
}