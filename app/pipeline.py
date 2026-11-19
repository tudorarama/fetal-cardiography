"""The whole method on one image: heart and thorax -> four chambers -> measurements -> flags.

    stage 1  src/outline.py      U-Net heart and thorax outline, quality checks, CTR
    stage 2  four_chambers/      nnU-Net v2 chambers in a square crop around the heart
    rules    src/anomaly.py      cardiomegaly, chamber disproportion, missing chamber

Every step can refuse the image instead of guessing: colour Doppler, no clear heart
(no chambers), heart and thorax not usable together (no CTR), implausible chamber
segmentation (no chamber rules). A refused step is reported with its reason.

Needs the four_chambers environment (CUDA PyTorch + nnU-Net v2 + MONAI):
    four_chambers\\.venv\\Scripts\\python.exe -m app.analyse <image>
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FOUR_CHAMBERS = ROOT / "four_chambers"
for p in (ROOT, FOUR_CHAMBERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config as fc_config  # noqa: E402  four_chambers/config.py
from common import ROI_SIZE, crop, roi_box, uncrop  # noqa: E402
from measurements import chamber_measurements, segmentation_quality  # noqa: E402

from src.anomaly import chambers_plausible, load_reference, screen  # noqa: E402
from src.config import MODELS_DIR  # noqa: E402
from src.outline import (DOPPLER_FRACTION, analyse, colour_fraction, heart_found,  # noqa: E402
                         predict_probs, verdict)
from src.unet import checkpoint_path, load_model  # noqa: E402

CHAMBERS = fc_config.CHAMBERS
CHAMBER_COLOURS = {"LV": (230, 40, 40), "RV": (40, 90, 230), "LA": (240, 200, 0), "RA": (40, 180, 70)}  # RGB
HEART_COLOUR, THORAX_COLOUR = (255, 140, 0), (220, 220, 220)


@dataclass
class Analysis:
    """Result of Pipeline.run on one image (all masks at the original image size)."""
    image: np.ndarray                         # grayscale uint8 (H, W)
    doppler: bool
    heart_ok: bool                            # heart outline clear enough for the chambers
    outline_quality: dict                     # src.outline.analyse measures (unet_pre)
    masks: dict                               # 'cardiac', 'thorax' ellipses 0/1 of the CTR model
    ctr: float | None                         # None when heart + thorax are not usable together
    ctr_reasons: str                          # why there is no CTR ('' = usable)
    roi_box: tuple | None = None              # (x0, y0, side) crop given to nnU-Net
    labels: np.ndarray | None = None          # chamber label map: 0 bg, 1 LV, 2 RV, 3 LA, 4 RA
    chambers: dict | None = None              # areas, fractions, lv_rv, la_ra
    chamber_quality: dict = field(default_factory=dict)  # coverage, solidity, layout_ok
    chambers_ok: bool = False                 # chamber measurements plausible -> chamber rules
    findings: list = field(default_factory=list)          # src.anomaly.Finding
    seconds: dict = field(default_factory=dict)

    @property
    def status(self):
        """'refused', 'flagged' or 'no flag'."""
        if self.doppler or (self.ctr is None and not self.chambers_ok):
            return "refused"
        return "flagged" if self.findings else "no flag"

    def summary(self):
        """JSON-friendly summary (no images)."""
        ch = self.chambers or {}
        return {
            "status": self.status,
            "doppler": self.doppler,
            "heart_found": self.heart_ok,
            "ctr": self.ctr,
            "ctr_refused_because": self.ctr_reasons or None,
            "chambers_plausible": self.chambers_ok,
            "lv_rv": ch.get("lv_rv"),
            "la_ra": ch.get("la_ra"),
            "chamber_fractions": ch.get("fractions"),
            "chamber_quality": self.chamber_quality or None,
            "findings": [{"code": f.code, "message": f.message, "value": f.value, "limit": f.limit}
                         for f in self.findings],
            "seconds": self.seconds,
        }


class Pipeline:
    """Loads the three trained models once; run(image) analyses one image."""

    def __init__(self, device=None, tta=True, trainer=None, fold=0, checkpoint="checkpoint_final.pth"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.outline_model, self.outline_cfg = load_model(checkpoint_path("pre"), self.device)
        self.ctr_model, self.ctr_cfg = load_model(checkpoint_path("pre_cm"), self.device)
        ref = MODELS_DIR / "anomaly_reference.json"
        self.reference = load_reference(ref) if ref.exists() else None
        self.chamber_model = _load_nnunet(self.device, tta, trainer, fold, checkpoint)

    # ---------------------------------------------------------------- steps
    def run(self, image):
        """image: path, BGR or grayscale array. Returns an Analysis."""
        bgr = _read(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        seconds, t0 = {}, time.perf_counter()

        doppler = colour_fraction(bgr) > DOPPLER_FRACTION
        masks, q = analyse(predict_probs(self.outline_model, self.outline_cfg, gray, self.device), gray.shape)
        masks_cm, q_cm = analyse(predict_probs(self.ctr_model, self.ctr_cfg, gray, self.device), gray.shape)
        reasons = "colour Doppler" if doppler else verdict(q_cm)
        # shown and measured: the CTR model's outlines; the chamber crop uses unet_pre, as in training
        a = Analysis(image=gray, doppler=doppler, heart_ok=not doppler and heart_found(q),
                     outline_quality=q, masks=masks_cm, ctr=None if reasons else q_cm["ctr"],
                     ctr_reasons=reasons)
        seconds["heart and thorax"] = time.perf_counter() - t0

        if a.heart_ok:
            t0 = time.perf_counter()
            heart = masks["cardiac"] > 0
            a.roi_box = roi_box(heart)
            a.labels = self._chambers(gray, a.roi_box)
            m = chamber_measurements(a.labels)
            a.chambers = {k: m[k] for k in ("areas", "fractions", "lv_rv", "la_ra")}
            a.chamber_quality = {"coverage": sum(m["areas"].values()) / max(heart.sum(), 1),
                                 **segmentation_quality(a.labels)}
            a.chambers_ok = chambers_plausible(**a.chamber_quality)
            seconds["four chambers"] = time.perf_counter() - t0

        meas = {"ctr_area": a.ctr}
        if a.chambers_ok:
            meas.update(lv_rv=a.chambers["lv_rv"], la_ra=a.chambers["la_ra"],
                        chamber_fractions=a.chambers["fractions"])
        if not a.doppler:
            a.findings = screen(meas, self.reference)
        a.seconds = {k: round(v, 2) for k, v in seconds.items()}
        return a

    def _chambers(self, gray, box):
        """nnU-Net probabilities in the crop -> back into the full image -> argmax (as predict.py)."""
        roi = crop(gray, box, ROI_SIZE, cv2.INTER_LINEAR).astype(np.float32)
        prob = self.chamber_model(roi)
        full = uncrop(np.ascontiguousarray(prob.transpose(1, 2, 0)), box, gray.shape)
        return full.argmax(-1).astype(np.uint8)


# ----------------------------------------------------------------- helpers
def _read(image):
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        return bgr
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def _load_nnunet(device, tta, trainer, fold, checkpoint):
    """predict(roi float32 (S, S)) -> probabilities (C, S, S) with the trained nnU-Net, in memory."""
    fc_config.setup_nnunet_env()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    last = fc_config.NNUNET_DIR / "last_trainer.txt"
    trainer = trainer or (last.read_text().strip() if last.exists() else None)
    folder = fc_config.NNUNET_DIR / "results" / fc_config.DATASET_NAME / \
        f"{trainer}__nnUNetPlans__{fc_config.CONFIGURATION}"
    if trainer is None or not (folder / f"fold_{fold}" / checkpoint).exists():
        raise FileNotFoundError(f"no trained four-chamber model in {folder} "
                                "(train it, or unzip the GitHub release in four_chambers/)")

    def make(dev):
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, device=dev,
                            perform_everything_on_device=dev.type == "cuda", verbose=False,
                            verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(folder), use_folds=(fold,), checkpoint_name=checkpoint)
        return p

    predictors = {device.type: make(device)}

    def predict(roi):
        x = roi[None, None]  # (channel, z=1, y, x), as nnU-Net reads a 2D PNG
        props = {"spacing": [999.0, 1.0, 1.0]}
        _, prob = predictors[device.type].predict_single_npy_array(x, props, None, None, True)
        prob = np.asarray(prob, dtype=np.float32)[:, 0]
        if np.isnan(prob).any():  # fp16 on GTX 16xx cards can give NaN: fall back to the CPU
            if "cpu" not in predictors:
                print("nnU-Net gave NaN on the GPU (fp16): this image and the next ones that do "
                      "are predicted on the CPU", file=sys.stderr)
                predictors["cpu"] = make(torch.device("cpu"))
            _, prob = predictors["cpu"].predict_single_npy_array(x, props, None, None, True)
            prob = np.asarray(prob, dtype=np.float32)[:, 0]
        return prob

    return predict


# ------------------------------------------------------------------ figures
def render(a, chambers_alpha=0.45):
    """RGB picture of the full image: thorax and heart outlines, chambers filled."""
    vis = cv2.cvtColor(a.image, cv2.COLOR_GRAY2RGB)
    width = max(2, a.image.shape[1] // 350)
    if a.labels is not None:
        fill = vis.copy()
        for i, c in enumerate(CHAMBERS, start=1):
            fill[a.labels == i] = CHAMBER_COLOURS[c]
        vis = cv2.addWeighted(fill, chambers_alpha, vis, 1 - chambers_alpha, 0)
        for i, c in enumerate(CHAMBERS, start=1):
            cnt, _ = cv2.findContours((a.labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnt, -1, CHAMBER_COLOURS[c], max(1, width // 2))
    for k, col in (("thorax", THORAX_COLOUR), ("cardiac", HEART_COLOUR)):
        cnt, _ = cv2.findContours(a.masks[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnt, -1, col, width)
    return vis


def render_zoom(a, size=420):
    """RGB close-up of the heart (the nnU-Net crop) with the chambers, or None."""
    if a.roi_box is None:
        return None
    return crop(render(a), a.roi_box, size, cv2.INTER_AREA)


def to_json(a):
    return json.dumps(a.summary(), indent=2, default=float)
