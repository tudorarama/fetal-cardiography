"""U-Net segmentation of the heart and thorax (MONAI), trained on FOCUS.

The network is trained here from random weights: MONAI only provides the
architecture and the loss.

Input: the image goes through src.preprocessing.prepare (median + bilateral
filter, CLAHE, letterbox to UNET_SIZE x UNET_SIZE, z-score). The probability
maps are mapped back with the inverse letterbox, so they are evaluated at the
original resolution with the same code as the classical method.

Output: 2 sigmoid channels, 0 = cardiac, 1 = thorax. They are independent
(not softmax) because the heart lies inside the thorax and the masks overlap.
"""
import cv2
import numpy as np
import torch
from monai.networks.nets import UNet

from src.augmentation import augment
from src.config import MODELS_DIR, PREPROC, UNET_SIZE
from src.data import STRUCTURES, list_cases, load_case
from src.postprocess import refine
from src.preprocessing import mask_to_unet, prepare, restore_size


def build_model():
    return UNet(spatial_dims=2, in_channels=1, out_channels=len(STRUCTURES),
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                num_res_units=2, dropout=0.1)


def preproc_config(use_filters=True):
    """Preprocessing settings; use_filters=False is the 'no preprocessing' ablation."""
    return {**PREPROC, "filter": use_filters}


def checkpoint_path(name):
    """models/unet_<name>.pth (pre = with filters, raw = without, _cm = heart-size augmentation)."""
    return MODELS_DIR / f"unet_{name}.pth"


# ---------------------------------------------------------------------- dataset

def load_split(split, cfg):
    """All cases of a split as a list of {'img': (1,S,S), 'mask': (2,S,S)} float32 arrays."""
    items = []
    for name in list_cases(split):
        img, _, gt = load_case(split, name)
        x = prepare(img, cfg=cfg).numpy()
        m = np.stack([mask_to_unet(gt[k], UNET_SIZE) for k in STRUCTURES])
        items.append({"img": x, "mask": m.astype(np.float32), "name": name})
    return items


def random_heart_size(img, mask, rng, s_range=(0.85, 1.4)):
    """Grows or shrinks the heart by a random factor (src.synthetic.enlarge_heart).

    FOCUS hearts all have a similar size relative to the thorax (CTR 0.15-0.36),
    so a network can learn "the heart is always about this big" and then
    underestimate enlarged hearts. Changing the heart size during training
    removes that shortcut.
    """
    from src.synthetic import enlarge_heart
    masks = {k: mask[i].astype(np.uint8) for i, k in enumerate(STRUCTURES)}
    if masks["cardiac"].sum() == 0:
        return img, mask
    w, wm = enlarge_heart(img[0], masks, float(rng.uniform(*s_range)))
    return w[None].astype(np.float32), np.stack([wm[k] for k in STRUCTURES]).astype(np.float32)


class FocusDataset(torch.utils.data.Dataset):
    """Training pairs; with train=True every access is randomly augmented (src.augmentation)."""

    def __init__(self, items, train=False, heart_size_aug=False, seed=0):
        self.items, self.train, self.heart_size_aug = items, train, heart_size_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]["img"], self.items[i]["mask"]
        if self.train:
            if self.heart_size_aug and self.rng.random() < 0.5:
                img, mask = random_heart_size(img, mask, self.rng)
            img, mask = augment(img, mask, rng=self.rng)
        return torch.as_tensor(np.ascontiguousarray(img)), torch.as_tensor(np.ascontiguousarray(mask))


# ------------------------------------------------------------------- inference

def load_model(path, device="cpu"):
    """The network and the preprocessing settings it was trained with."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt.get("preproc", PREPROC)


def make_predictor(path, device="cpu", threshold=0.5):
    """predict(img) -> (masks, ellipses), the interface used by src.evaluation.

    The probability maps are mapped back to the original size, thresholded and
    refined with the SAME post-processing as the classical method (largest
    component, hole filling, ellipse fitting).
    """
    model, cfg = load_model(path, device)

    @torch.no_grad()
    def predict(img):
        x, meta = prepare(img, cfg=cfg, return_meta=True)
        prob = torch.sigmoid(model(x[None].to(device)))[0].cpu().numpy()
        masks, ellipses = {}, {}
        for c, k in enumerate(STRUCTURES):
            p = restore_size(prob[c], meta["resize"], interp=cv2.INTER_LINEAR)
            masks[k], ellipses[k] = refine((p > threshold).astype(np.uint8))
        return masks, ellipses

    return predict
