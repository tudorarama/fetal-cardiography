"""Segment Anything (SAM) as an annotation ASSISTANT for the chamber annotation tool.

SAM (Kirillov et al., 2023, Meta AI) is a general "promptable" segmentation
model: given an image and a few clicks or a box it returns a mask, but it has
no idea what a ventricle is. Here it only speeds up the drawing; the annotator
decides which structure each mask is, corrects it, and accepts it.

Observed on FOCUS: for a single click SAM's highest-scoring proposal is often
the whole image (ultrasound has weak edges), while the smallest proposal is
usually the cavity that was clicked. Hence:
  * proposals are sorted by size and the smallest plausible one is shown first,
  * a box prompt (drag) and negative clicks are supported to stop leaks,
  * the mask is clipped to an allowed region (chambers: inside the FOCUS heart
    outline, slightly dilated; spine: inside the dilated thorax).

Weights: ../models/sam_vit_b_01ec64.pth (config.SAM_WEIGHTS) (ViT-B, ~360 MB), from
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from config import SAM_WEIGHTS
WEIGHTS = SAM_WEIGHTS
MAX_FRACTION = 0.3   # proposals larger than this fraction of the view are ignored
MAX_ALLOWED = 0.6   # ... or larger than this fraction of the allowed region (heart / thorax)
MIN_PIXELS = 30


def available(weights=WEIGHTS):
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        return False
    return Path(weights).exists()


class SamAssistant:
    """Wraps a SamPredictor with an embedding cache and background prefetching."""

    def __init__(self, weights=WEIGHTS):
        from segment_anything import SamPredictor, sam_model_registry
        torch.set_grad_enabled(False)
        self.model = sam_model_registry["vit_b"](checkpoint=str(weights)).eval()
        self.predictor = SamPredictor(self.model)
        self._make_predictor = lambda: SamPredictor(self.model)
        self.cache, self.lock, self.threads = {}, threading.Lock(), {}

    # ------------------------------------------------------------- embeddings
    def _compute(self, rgb):
        p = self._make_predictor()
        p.set_image(rgb)
        return p.features, p.original_size, p.input_size

    def prefetch(self, key, rgb):
        """Starts computing the embedding of an image in the background."""
        with self.lock:
            if key in self.cache or key in self.threads:
                return

        def work():
            emb = self._compute(rgb)
            with self.lock:
                self.cache[key] = emb
                self.threads.pop(key, None)
                if len(self.cache) > 6:  # keep memory bounded
                    self.cache.pop(next(iter(self.cache)))

        t = threading.Thread(target=work, daemon=True)
        with self.lock:
            self.threads[key] = t
        t.start()

    def set_image(self, key, rgb):
        """Makes the image current (blocks only if its embedding is not ready yet)."""
        with self.lock:
            emb, t = self.cache.get(key), self.threads.get(key)
        if emb is None and t is not None:
            t.join()
            with self.lock:
                emb = self.cache.get(key)
        if emb is None:
            emb = self._compute(rgb)
            with self.lock:
                self.cache[key] = emb
        p = self.predictor
        p.reset_image()
        p.features, p.original_size, p.input_size = emb
        p.is_image_set = True

    # ------------------------------------------------------------- prediction
    def proposals(self, points, labels, box=None, allowed=None):
        """Candidate masks (bool, view size) for the prompts, smallest plausible first.

        points: list of (x, y) in view pixels, labels: 1 positive / 0 negative,
        box: (x0, y0, x1, y1) or None, allowed: bool mask the result is clipped to.
        """
        kw = {"multimask_output": box is None or len(points) == 0}
        if points:
            kw["point_coords"] = np.array(points, dtype=np.float32)
            kw["point_labels"] = np.array(labels)
        if box is not None:
            kw["box"] = np.array(box, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(**kw)
        pos = [p for p, l in zip(points, labels) if l == 1]
        # a proposal that fills (almost) the whole allowed region is "everything", not a structure
        limit = MAX_FRACTION * masks[0].size
        if allowed is not None:
            limit = min(limit, MAX_ALLOWED * allowed.sum())
        out = []
        for m in masks:
            m = m.astype(bool)
            if allowed is not None:
                m &= allowed
            m = _clean(m, pos)
            duplicate = any((m & o).sum() > 0.95 * max(m.sum(), o.sum()) for o in out)
            if MIN_PIXELS <= m.sum() <= limit and not duplicate:
                out.append(m)
        out.sort(key=lambda m: m.sum())
        return out


def _clean(mask, positive_points):
    """Keeps the component containing the first positive click (else the largest), fills holes."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return mask
    keep = 0
    for x, y in positive_points:
        lab = cc[int(round(y)), int(round(x))] if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] else 0
        if lab:
            keep = lab
            break
    if not keep:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (cc == keep).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnt, -1, 1, -1)
    return filled.astype(bool)


def mask_to_polygon(mask, epsilon=1.0):
    """Largest outer contour of a mask as a list of (x, y), simplified by epsilon pixels."""
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return []
    c = cv2.approxPolyDP(max(cnt, key=cv2.contourArea), epsilon, True)
    return [(float(x), float(y)) for x, y in c[:, 0]]
