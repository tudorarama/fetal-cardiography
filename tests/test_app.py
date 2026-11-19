"""End-to-end demo pipeline (app/). The full run needs the four_chambers environment
(nnU-Net v2 + trained model) and FOCUS; it is skipped elsewhere."""
import numpy as np
import pytest

from src.anomaly import chambers_plausible
from src.outline import DOPPLER_FRACTION, colour_fraction


def test_colour_fraction_grey_vs_doppler():
    grey = np.full((50, 50, 3), 120, np.uint8)
    assert colour_fraction(grey) == 0.0
    doppler = grey.copy()
    doppler[:10, :10] = (0, 0, 255)  # 4 % red pixels
    assert colour_fraction(doppler) > DOPPLER_FRACTION


@pytest.mark.parametrize("coverage, solidity, layout, ok", [
    (0.6, 0.99, True, True),
    (0.6, 0.99, None, True),     # layout unknown (a chamber missing): not a reason to refuse
    (0.1, 0.99, True, False),    # chambers cover little of the heart
    (0.6, 0.70, True, False),    # a chamber split into pieces
    (0.6, 0.99, False, False),   # impossible arrangement
])
def test_chambers_plausible(coverage, solidity, layout, ok):
    assert chambers_plausible(coverage, solidity, layout) is ok


def test_pipeline_on_a_focus_test_image():
    pytest.importorskip("nnunetv2")
    pytest.importorskip("monai")
    from app.pipeline import FOUR_CHAMBERS, Pipeline, render, render_zoom
    from src.data import FOCUS_DIR
    image = FOCUS_DIR / "testing" / "images" / "001.png"
    if not image.exists() or not (FOUR_CHAMBERS / "nnunet" / "last_trainer.txt").exists():
        pytest.skip("FOCUS or the trained four-chamber model is missing")

    a = Pipeline().run(image)
    assert a.heart_ok and a.chambers_ok and a.ctr is not None
    assert 0.15 < a.ctr < 0.35
    assert a.labels.shape == a.image.shape and set(np.unique(a.labels)) == {0, 1, 2, 3, 4}
    assert abs(sum(a.chambers["fractions"].values()) - 1) < 1e-6
    assert a.status == "no flag"
    assert render(a).shape == (*a.image.shape, 3)
    assert render_zoom(a).shape[2] == 3
