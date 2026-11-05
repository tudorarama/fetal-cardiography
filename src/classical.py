"""Classical segmentation baseline: radial search + Otsu for the thorax,
hierarchical Chan-Vese for the heart.

All processing runs on a downscaled copy of the image (longest side
WORK_SIZE px) for speed; the resulting ellipses are scaled back to the original
resolution. Kernel sizes below are therefore in "work" pixels.

Pipeline
    1. field of view: the ultrasound sector, without the black background,
       the on-screen text and the grey-scale bar;
    2. thorax: rays cast from the expected thorax centre; on each ray the
       thoracic wall is the innermost strong bright->dark edge whose inner side
       is above the Otsu threshold; robust ellipse fit, repeated from the
       fitted centre.  (A region-based Otsu + morphology variant is kept in
       segment_thorax_region: it does worse than a fixed circle, because the
       thorax touches other bright tissue in most images.)
    3. heart: Chan-Vese applied hierarchically inside the thorax keeps only
       the darkest phase (the blood in the chambers) -> the chamber blobs
       closest to the thorax centre are merged -> convex hull -> ellipse,
       enlarged by a factor calibrated on the training set (the GT heart also
       contains the myocardium around the chambers).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import chan_vese

from src.data import ellipse_to_mask
from src.postprocess import fill_holes, fit_ellipse, largest_component

WORK_SIZE = 480


@dataclass
class ClassicalParams:
    # field of view
    fov_threshold: int = 8
    fov_open: int = 9
    thorax_method: str = "radial"   # "radial" (segment_thorax) or "region"
    # thorax, region-based variant (segment_thorax_region)
    thorax_sigma: float = 6.0
    thorax_close: int = 15
    thorax_open: int = 25
    # thorax, radial search (segment_thorax); priors measured on the FOCUS training set
    seed_x: float = 0.492           # mean thorax centre / image width
    seed_y: float = 0.494           # mean thorax centre / image height
    prior_radius: float = 0.359     # mean sqrt(a*b) / min(image height, width)
    prior_spread: float = 0.25      # relative std of the radius prior, first pass
    prior_spread_refine: float = 0.2
    radial_sigma: float = 5.0
    n_rays: int = 90
    wall_offset: int = 4            # px inside the edge that must be brighter than Otsu
    fit_keep: float = 0.8
    radial_passes: int = 2
    wall_pick: float = 0.7          # 1.0 = strongest edge; lower = innermost edge this strong
    # heart
    heart_inner: float = 0.85       # search the heart inside this fraction of the thorax ellipse
    cv_mu: float = 0.25
    cv_iters: int = 150
    cv_levels: int = 3              # hierarchical Chan-Vese levels
    cv_sigma: float = 1.5           # Gaussian smoothing of the crop before Chan-Vese
    heart_open: int = 5
    heart_close: int = 15
    heart_min_area: float = 0.01    # dark blobs smaller than this fraction of the thorax are ignored
    heart_cluster: float = 0.5      # blobs within this fraction of the thorax minor semi-axis join the main one
    heart_center_spread: float = 0.4  # prior on heart distance from thorax centre (x thorax radius)
    heart_scale: float = 1.0        # axis scale factor, calibrated on train


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _scale_ellipse(e, s):
    cx, cy, a, b, ang = e
    return (cx * s, cy * s, a * s, b * s, ang)


def field_of_view(img, p):
    """Binary mask of the ultrasound sector (removes background, text, scale bar)."""
    m = (cv2.medianBlur(img, 5) > p.fov_threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(p.fov_open))
    return fill_holes(largest_component(m))


def _closest_to_center(mask, center):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    dist = np.linalg.norm(centroids[1:] - np.asarray(center), axis=1)
    # large components close to the centre win
    score = areas / (1.0 + dist) ** 2
    return (labels == 1 + int(np.argmax(score))).astype(np.uint8)


def segment_thorax_region(work, fov, p):
    """Region-based variant (kept for comparison): Otsu -> morphology -> convex hull.

    Fails when the thorax touches other bright tissue, which is the common case.
    """
    blur = cv2.GaussianBlur(work, (0, 0), p.thorax_sigma)
    t = threshold_otsu(blur[fov > 0])
    m = ((blur > t) & (fov > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(p.thorax_close))
    m = fill_holes(m)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(p.thorax_open))
    ys, xs = np.nonzero(fov)
    m = _convex_hull(_closest_to_center(m, (xs.mean(), ys.mean())))
    return m, fit_ellipse(m)


def _ray_profiles(img, center, radii, angles):
    """Intensity sampled along rays: array (n_angles, n_radii) plus the sample coordinates."""
    xs = center[0] + np.outer(np.cos(angles), radii)
    ys = center[1] + np.outer(np.sin(angles), radii)
    prof = cv2.remap(img.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return prof, xs, ys


def _robust_fit_ellipse(pts, iters=3, keep=0.8):
    """Ellipse fit that repeatedly drops the points with the largest residual."""
    pts = np.asarray(pts, np.float32)
    e = None
    for _ in range(iters + 1):
        if len(pts) < 5:
            break
        (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
        if w <= 0 or h <= 0:
            break
        e = (cx, cy, w / 2, h / 2, ang)
        t = np.deg2rad(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        resid = np.abs(np.sqrt((u / e[2]) ** 2 + (v / e[3]) ** 2) - 1)
        pts = pts[resid <= np.quantile(resid, keep)]
    return e


def segment_thorax(work, fov, p):
    """Thorax ellipse by radial search for the thoracic wall (work resolution).

    From a seed point, rays are cast in all directions. Along each ray the wall
    is the outer edge of the bright ring (ribs, skin): the place where the
    smoothed intensity falls from above the Otsu threshold to below it, with the
    strongest drop, weighted by a prior on the thorax radius learnt on the
    training set. A robust ellipse fit on these points gives the thorax. The
    search is repeated from the centre of the first ellipse.
    """
    h, w = work.shape
    blur = cv2.GaussianBlur(work, (0, 0), p.radial_sigma).astype(np.float32)
    otsu = threshold_otsu(blur[fov > 0])
    fov_er = cv2.erode(fov, _kernel(7))

    center = np.array([p.seed_x * w, p.seed_y * h])
    r0 = p.prior_radius * min(h, w)
    spread = p.prior_spread
    angles = np.linspace(0, 2 * np.pi, p.n_rays, endpoint=False)
    e = None
    for _ in range(p.radial_passes):
        radii = np.arange(0.4 * r0, 1.8 * r0, 1.0)
        prof, xs, ys = _ray_profiles(blur, center, radii, angles)
        inside, _, _ = _ray_profiles(fov_er, center, radii, angles)
        drop = -np.gradient(prof, axis=1)                       # bright -> dark going outwards
        drop = cv2.GaussianBlur(drop, (0, 0), sigmaX=2, sigmaY=0.01)
        # a bright wall just inside the edge: compare with the sample wall_offset px closer
        # to the centre (no wrap-around, the first samples have nothing inside them)
        bright_inside = np.zeros_like(prof, dtype=bool)
        bright_inside[:, p.wall_offset:] = prof[:, :-p.wall_offset] > otsu
        prior = np.exp(-0.5 * ((radii - r0) / (spread * r0)) ** 2)
        score = np.where((inside > 0) & bright_inside, drop, 0) * prior
        pts = []
        for i in range(len(angles)):
            best = score[i].max()
            if best <= 0:
                continue
            # innermost edge that is almost as strong as the best one: the thoracic
            # wall, not the skin or other tissue further out
            j = int(np.argmax(score[i] >= p.wall_pick * best))
            pts.append((xs[i, j], ys[i, j]))
        if len(pts) < 8:
            break
        e = _robust_fit_ellipse(pts, keep=p.fit_keep)
        if e is None:
            break
        center = np.array(e[:2])
        r0, spread = np.sqrt(e[2] * e[3]), p.prior_spread_refine
    if e is None:
        return np.zeros_like(fov), None
    return ellipse_to_mask(work.shape, e), e


def _convex_hull(mask):
    """Filled convex hull of a mask: closes concavities before ellipse fitting."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out


def segment_heart(work, thorax_ellipse, p):
    """Heart mask (raw, before the scale factor) and ellipse in work resolution."""
    cx, cy, a, b, ang = thorax_ellipse
    inner = ellipse_to_mask(work.shape, (cx, cy, a * p.heart_inner, b * p.heart_inner, ang))
    x, y, w, h = cv2.boundingRect(inner)
    crop = work[y:y + h, x:x + w].astype(float) / 255.0
    roi = inner[y:y + h, x:x + w] > 0

    if p.cv_sigma:
        crop = cv2.GaussianBlur(crop, (0, 0), p.cv_sigma)

    # Hierarchical Chan-Vese: level 1 separates bright tissue (ribs, spine,
    # myocardium) from the rest; each further level splits the dark phase again,
    # until only the darkest region - the blood in the chambers - remains.
    dark = roi.copy()
    for _ in range(p.cv_levels):
        img = crop.copy()
        # pixels already excluded become bright, so they join the bright phase
        img[~dark] = np.percentile(crop[dark], 90)
        seg = chan_vese(img, mu=p.cv_mu, lambda1=1, lambda2=1, max_num_iter=p.cv_iters,
                        init_level_set="checkerboard")
        a_in, b_in = seg & dark, ~seg & dark
        if not a_in.any() or not b_in.any():
            return None, None
        dark = a_in if crop[a_in].mean() < crop[b_in].mean() else b_in
    dark = dark.astype(np.uint8)

    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _kernel(p.heart_open))
    # chambers: the largest dark blob plus the dark blobs close to it
    # (the four chambers are separated only by the thin septum and valves)
    min_area = p.heart_min_area * roi.sum()
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not blobs:
        return None, None
    # main chamber blob: large and close to the thorax centre
    tc = np.array([cx - x, cy - y])
    r = np.sqrt(a * b)

    def blob_score(i):
        d = np.linalg.norm(centroids[i] - tc) / r
        return stats[i, cv2.CC_STAT_AREA] * np.exp(-0.5 * (d / p.heart_center_spread) ** 2)

    main = max(blobs, key=blob_score)
    reach = p.heart_cluster * min(a, b)
    keep = [i for i in blobs if np.linalg.norm(centroids[i] - centroids[main]) <= reach]
    dark = np.isin(labels, keep).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _kernel(p.heart_close))
    dark = _convex_hull(dark)

    full = np.zeros(work.shape, np.uint8)
    full[y:y + h, x:x + w] = dark
    e = fit_ellipse(full)
    return full, e


@dataclass
class ClassicalResult:
    thorax: tuple = None            # ellipse (cx, cy, a, b, angle) in original pixels
    heart: tuple = None
    heart_unscaled: tuple = None    # heart ellipse before heart_scale (for calibration)
    masks: dict = field(default_factory=dict)   # 'cardiac', 'thorax' -> uint8 masks
    debug: dict = field(default_factory=dict)   # intermediate work-resolution images


def segment(pre, p=None, keep_debug=False):
    """Runs the classical pipeline on a preprocessed grayscale image."""
    p = p or ClassicalParams()
    h0, w0 = pre.shape
    s = WORK_SIZE / max(h0, w0)
    work = cv2.resize(pre, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)

    res = ClassicalResult()
    empty = np.zeros(pre.shape, np.uint8)
    res.masks = {"cardiac": empty, "thorax": empty.copy()}

    fov = field_of_view(work, p)
    find_thorax = segment_thorax if p.thorax_method == "radial" else segment_thorax_region
    thorax_mask, thorax_e = find_thorax(work, fov, p)
    if keep_debug:
        res.debug.update(work=work, fov=fov, thorax_raw=thorax_mask)
    if thorax_e is None:
        return res
    res.thorax = _scale_ellipse(thorax_e, 1 / s)
    res.masks["thorax"] = ellipse_to_mask(pre.shape, res.thorax)

    heart_mask, heart_e = segment_heart(work, thorax_e, p)
    if keep_debug:
        res.debug["heart_raw"] = heart_mask
    if heart_e is None:
        return res
    res.heart_unscaled = _scale_ellipse(heart_e, 1 / s)
    cx, cy, a, b, ang = heart_e
    heart_e = (cx, cy, a * p.heart_scale, b * p.heart_scale, ang)
    res.heart = _scale_ellipse(heart_e, 1 / s)
    # the heart lies inside the thorax
    res.masks["cardiac"] = ellipse_to_mask(pre.shape, res.heart) & res.masks["thorax"]
    return res


# ------------------------------------------------------------------ interface

def make_predictor(p=None, preprocess_kwargs=None):
    """predict(img) -> (masks, ellipses), the interface used by src.evaluation."""
    from src.preprocess import preprocess

    p = p or ClassicalParams()
    pk = preprocess_kwargs or {}

    def predict(img):
        r = segment(preprocess(img, **pk), p)
        return r.masks, {"cardiac": r.heart, "thorax": r.thorax}

    return predict


def save_params(p, path):
    Path(path).write_text(json.dumps(asdict(p), indent=2))


def load_params(path):
    return ClassicalParams(**json.loads(Path(path).read_text()))
