"""Checks that the ablation variants switch the preprocessing steps as intended."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from ablation_preprocessing import VARIANTS, variant_cfg
from src.config import PREPROC
from src.preprocessing import preprocess


def _fake_image(h=200, w=300):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_none_leaves_the_image_unchanged():
    img = _fake_image()
    assert np.array_equal(preprocess(img, variant_cfg("none")), img)


def test_full_is_the_project_pipeline():
    img = _fake_image()
    assert np.array_equal(preprocess(img, variant_cfg("full")), preprocess(img, PREPROC))


def test_every_variant_is_different():
    img = _fake_image()
    outs = {v: preprocess(img, variant_cfg(v)) for v in VARIANTS}
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(outs[a], outs[b]), (a, b)


def main():
    for fn in (test_none_leaves_the_image_unchanged, test_full_is_the_project_pipeline,
               test_every_variant_is_different):
        fn()
        print(f"ok  {fn.__name__}")


if __name__ == "__main__":
    main()