"""Synthetic cardiomegaly: enlarges the heart inside the thorax by warping the image.

FOCUS contains almost no enlarged hearts (ground-truth CTR 0.15-0.36), so a
cardiomegaly detector cannot be tested on it directly. This module creates
"what if" images: a radial warp centred on the heart grows the heart by a
linear factor s (area by about s^2) and compresses the tissue between the heart
and the thoracic wall (the lungs), while the thorax itself is left unchanged.
The ground-truth masks are warped with the same map, so the new CTR is known.

It is a stress test of the measurement chain, not a realistic simulation of a
diseased heart: the enlarged heart is just a magnified copy of the normal one.
"""
import cv2
import numpy as np

N_DIRECTIONS = 360


def _boundary_radius(mask, center, max_r, step=0.5):
    """Distance from center to the mask boundary along N_DIRECTIONS rays."""
    theta = np.linspace(-np.pi, np.pi, N_DIRECTIONS, endpoint=False)
    r = np.arange(0, max_r, step)
    xs = center[0] + np.outer(np.cos(theta), r)
    ys = center[1] + np.outer(np.sin(theta), r)
    inside = np.zeros(xs.shape, dtype=bool)
    ok = (xs >= 0) & (ys >= 0) & (xs < mask.shape[1] - 1) & (ys < mask.shape[0] - 1)
    inside[ok] = mask[ys[ok].round().astype(int), xs[ok].round().astype(int)] > 0
    first_out = np.argmax(~inside, axis=1)  # first sample outside the mask
    first_out[inside.all(axis=1)] = len(r) - 1
    return theta, r[first_out]


def enlarge_heart(img, masks, s, margin=0.95):
    """Returns (warped image, warped masks) with the heart grown by factor s.

    masks: {'cardiac', 'thorax'} 0/1 arrays. In each direction the heart edge
    moves from r_H to s * r_H, but never past margin * r_T (thoracic wall).
    """
    heart, thorax = masks["cardiac"], masks["thorax"]
    m = cv2.moments(heart, binaryImage=True)
    c = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    max_r = float(np.hypot(*img.shape[:2]))
    theta, r_h = _boundary_radius(heart, c, max_r)
    _, r_t = _boundary_radius(thorax, c, max_r)

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - c[0], yy - c[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    rh = np.interp(phi, theta, r_h, period=2 * np.pi)
    rt = np.interp(phi, theta, r_t, period=2 * np.pi)
    new_rh = np.minimum(s * rh, margin * rt)  # new heart edge, kept inside the thorax

    # inverse map: output radius r -> source radius r_src
    r_src = r.copy()
    inner = r <= new_rh
    r_src[inner] = r[inner] * rh[inner] / np.maximum(new_rh[inner], 1e-6)
    ring = ~inner & (r < rt)
    r_src[ring] = rh[ring] + (r[ring] - new_rh[ring]) * (rt[ring] - rh[ring]) / \
        np.maximum(rt[ring] - new_rh[ring], 1e-6)

    scale = np.where(r > 0, r_src / np.maximum(r, 1e-6), 1.0)
    map_x = (c[0] + dx * scale).astype(np.float32)
    map_y = (c[1] + dy * scale).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out_masks = {k: cv2.remap(v, map_x, map_y, cv2.INTER_NEAREST) for k, v in masks.items()}
    return out, out_masks
