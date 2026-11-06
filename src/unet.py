"""U-Net segmentation of the heart and thorax (MONAI), trained on FOCUS.

The network is trained here from random weights: MONAI only provides the
architecture, the losses and the augmentations.

Geometry: every image is letterboxed (zero padding to a square, image in the
top-left corner) and resized to SIZE x SIZE. Predictions are mapped back with the
inverse transform, so they are evaluated at the original resolution with the
same code as the classical method.

Output: 2 sigmoid channels, 0 = cardiac, 1 = thorax. They are independent
(not softmax) because the heart lies inside the thorax and the masks overlap.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.networks.nets import UNet
from monai.transforms import (Compose, RandAdjustContrastd, RandAffined, RandFlipd,
                              RandGaussianNoised, RandScaleIntensityd)

from src.data import STRUCTURES, list_cases, load_case
from src.postprocess import refine
from src.preprocess import preprocess

SIZE = 256


# --------------------------------------------------------------------- geometry

def letterbox(img, size=SIZE, interpolation=cv2.INTER_AREA):
    """Pads to a square (top-left aligned) and resizes. Returns the image and the side length."""
    h, w = img.shape[:2]
    side = max(h, w)
    sq = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    sq[:h, :w] = img
    return cv2.resize(sq, (size, size), interpolation=interpolation), side


def unletterbox(arr, shape, side):
    """Inverse of letterbox for a (size, size) float map -> original (H, W)."""
    big = cv2.resize(arr, (side, side), interpolation=cv2.INTER_LINEAR)
    return big[:shape[0], :shape[1]]


# ---------------------------------------------------------------------- dataset

def build_model():
    return UNet(spatial_dims=2, in_channels=1, out_channels=len(STRUCTURES),
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                num_res_units=2, dropout=0.1)


def prepare(img, use_preprocess=True):
    """Image -> float32 network input (1, SIZE, SIZE) in [0, 1], plus the letterbox side."""
    x = preprocess(img) if use_preprocess else img
    x, side = letterbox(x)
    return x.astype(np.float32)[None] / 255.0, side


def load_split_tensors(split, use_preprocess=True):
    """All cases of a split as a list of {'img': (1,S,S), 'mask': (2,S,S)} float arrays."""
    items = []
    for name in list_cases(split):
        img, _, gt = load_case(split, name)
        x, _ = prepare(img, use_preprocess)
        m = np.stack([letterbox(gt[k], interpolation=cv2.INTER_NEAREST)[0] for k in STRUCTURES])
        items.append({"img": x, "mask": m.astype(np.float32), "name": name})
    return items


class RandHeartSized:
    """Random synthetic change of the heart size (src.synthetic.enlarge_heart).

    FOCUS hearts all have a similar size relative to the thorax (CTR 0.15-0.36),
    so a network can learn "the heart is always about this big" and then
    underestimate enlarged hearts. Growing or shrinking the heart by a random
    factor s during training removes that shortcut.
    """

    def __init__(self, prob=0.5, s_range=(0.85, 1.4)):
        self.prob, self.s_range = prob, s_range

    def __call__(self, d):
        if np.random.random() >= self.prob:
            return d
        from src.synthetic import enlarge_heart
        img = np.asarray(d["img"])[0]
        m = np.asarray(d["mask"]).astype(np.uint8)
        masks = {k: m[i] for i, k in enumerate(STRUCTURES)}
        if masks["cardiac"].sum() == 0:
            return d
        w, wm = enlarge_heart(img, masks, float(np.random.uniform(*self.s_range)))
        return {**d, "img": w[None].astype(np.float32),
                "mask": np.stack([wm[k] for k in STRUCTURES]).astype(np.float32)}


def train_transforms(heart_size_aug=False):
    """Augmentations applied identically to image and mask (except intensity ones)."""
    keys = ["img", "mask"]
    size = [RandHeartSized()] if heart_size_aug else []
    return Compose(size + [
        # left-right mirror only: an upside-down sector (probe at the bottom) never occurs
        RandFlipd(keys, prob=0.5, spatial_axis=1),
        RandAffined(keys, prob=0.8, rotate_range=np.pi / 6, scale_range=0.15,
                    translate_range=(20, 20), mode=("bilinear", "nearest"),
                    padding_mode="zeros"),
        RandScaleIntensityd("img", factors=0.2, prob=0.5),
        RandAdjustContrastd("img", gamma=(0.7, 1.5), prob=0.5),
        RandGaussianNoised("img", std=0.03, prob=0.3),
    ])


class FocusDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform=None):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = {"img": self.items[i]["img"], "mask": self.items[i]["mask"]}
        if self.transform is not None:
            d = self.transform(d)
        return torch.as_tensor(np.asarray(d["img"])), torch.as_tensor(np.asarray(d["mask"]))


# ------------------------------------------------------------------- inference

def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt.get("use_preprocess", True)


def make_predictor(path, device="cpu", threshold=0.5):
    """predict(img) -> (masks, ellipses), the interface used by src.evaluation.

    The probability maps are thresholded, mapped back to the original size and
    refined with the SAME post-processing as the classical method (largest
    component, hole filling, ellipse fitting).
    """
    model, use_preprocess = load_model(path, device)

    @torch.no_grad()
    def predict(img):
        x, side = prepare(img, use_preprocess)
        prob = torch.sigmoid(model(torch.from_numpy(x)[None].to(device)))[0].cpu().numpy()
        masks, ellipses = {}, {}
        for c, k in enumerate(STRUCTURES):
            m = (unletterbox(prob[c], img.shape, side) > threshold).astype(np.uint8)
            masks[k], ellipses[k] = refine(m)
        return masks, ellipses

    return predict


def checkpoint_path(name):
    return Path(__file__).resolve().parent.parent / "results" / f"unet_{name}.pth"
