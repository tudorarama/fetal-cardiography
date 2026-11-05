"""Ultrasound preprocessing: speckle reduction and contrast enhancement.

Pipeline: median filter -> bilateral filter -> CLAHE. Every step can be turned
off (set its main parameter to 0 / None) so the ablation "with vs without
preprocessing" is easy to run.
"""
import cv2


def median(img, k=5):
    """Removes isolated speckle points. k must be odd."""
    return cv2.medianBlur(img, k) if k else img


def bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """Smooths homogeneous regions while keeping edges sharp."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space) if d else img


def clahe(img, clip=2.0, grid=8):
    """Contrast Limited Adaptive Histogram Equalization (local contrast)."""
    if not clip:
        return img
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def preprocess(img, median_k=5, bilateral_d=9, sigma_color=75, sigma_space=75,
               clahe_clip=2.0, clahe_grid=8):
    """Full pipeline on a grayscale uint8 image."""
    out = median(img, median_k)
    out = bilateral(out, bilateral_d, sigma_color, sigma_space)
    return clahe(out, clahe_clip, clahe_grid)


def preprocess_steps(img, **kwargs):
    """Intermediate images of the pipeline, for figures: list of (title, image)."""
    p = dict(median_k=5, bilateral_d=9, sigma_color=75, sigma_space=75,
             clahe_clip=2.0, clahe_grid=8)
    p.update(kwargs)
    m = median(img, p["median_k"])
    b = bilateral(m, p["bilateral_d"], p["sigma_color"], p["sigma_space"])
    c = clahe(b, p["clahe_clip"], p["clahe_grid"])
    return [("original", img), (f"median k={p['median_k']}", m),
            (f"bilateral d={p['bilateral_d']}", b), (f"CLAHE clip={p['clahe_clip']}", c)]
