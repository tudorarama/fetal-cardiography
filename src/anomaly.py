"""Rule-based screening for congenital heart anomalies from the measurements.

No public dataset with anomaly labels and masks is available, so the detector
is a set of transparent rules on measurements that clinicians use on the
four-chamber view. Each rule compares a measurement with
  * a threshold from the literature (when one exists), and
  * the normal range of the FOCUS training set (z-score), used as reference.

A flag is a reason to look again ("screen positive"), never a diagnosis.

Rules
-----
cardiomegaly           CTR (area) > 0.35. Normal heart area is about 1/4 to 1/3
                       of the thorax; a large heart accompanies heart failure,
                       Ebstein anomaly, tricuspid dysplasia, anaemia...
ctr_outside_reference  |z| > 2.5 with respect to the FOCUS training set.
abnormal_axis          cardiac axis outside 25-65 degrees (normal 45 +- 20).
                       Needs the septum (chamber segmentation) and the spine
                       position, which is only available when it was annotated.
ventricle_disproportion LV/RV area ratio outside [0.5, 2]: hypoplastic left or
                       right heart, coarctation (small LV), ...
atrium_disproportion   LA/RA area ratio outside [0.5, 2].
missing_chamber        a chamber smaller than 3 % of the heart: single
                       ventricle, severe hypoplasia, AV septal defect (a large
                       defect merges the chambers).
The chamber thresholds are heuristic starting points; they should be replaced
by the range of the annotated normal images once enough are available.
"""
import json
from dataclasses import asdict, dataclass

CTR_LIMIT = 0.35
Z_LIMIT = 2.5
AXIS_RANGE = (25.0, 65.0)
RATIO_RANGE = (0.5, 2.0)
MIN_CHAMBER_FRACTION = 0.03


@dataclass
class Finding:
    code: str
    message: str
    value: float
    limit: str


def build_reference(ctr_values):
    """Normal range (mean, std) of the CTR, from ground truth of the training split."""
    import numpy as np
    v = np.asarray(ctr_values, dtype=float)
    return {"ctr_mean": float(v.mean()), "ctr_std": float(v.std(ddof=1)), "n": int(len(v))}


def save_reference(ref, path):
    with open(path, "w") as f:
        json.dump(ref, f, indent=2)


def load_reference(path):
    with open(path) as f:
        return json.load(f)


def screen(meas, reference=None):
    """meas: dict with any of ctr_area, axis, lv_rv, la_ra, chamber_fractions.

    Missing measurements are skipped. Returns a list of Finding (empty = no flag).
    """
    out = []
    ctr = meas.get("ctr_area")
    if ctr is not None:
        if ctr > CTR_LIMIT:
            out.append(Finding("cardiomegaly", "heart too large for the thorax",
                               ctr, f"> {CTR_LIMIT}"))
        if reference:
            z = (ctr - reference["ctr_mean"]) / reference["ctr_std"]
            if abs(z) > Z_LIMIT:
                out.append(Finding("ctr_outside_reference",
                                   f"CTR {'above' if z > 0 else 'below'} the normal range",
                                   z, f"|z| > {Z_LIMIT}"))
    axis = meas.get("axis")
    if axis is not None and not AXIS_RANGE[0] <= axis <= AXIS_RANGE[1]:
        out.append(Finding("abnormal_axis", "cardiac axis outside the normal range",
                           axis, f"{AXIS_RANGE[0]:.0f}-{AXIS_RANGE[1]:.0f} deg"))
    for key, code, label in (("lv_rv", "ventricle_disproportion", "LV/RV"),
                             ("la_ra", "atrium_disproportion", "LA/RA")):
        v = meas.get(key)
        if v is not None and not RATIO_RANGE[0] <= v <= RATIO_RANGE[1]:
            out.append(Finding(code, f"{label} area ratio out of range", v,
                               f"{RATIO_RANGE[0]}-{RATIO_RANGE[1]}"))
    for name, frac in (meas.get("chamber_fractions") or {}).items():
        if frac < MIN_CHAMBER_FRACTION:
            out.append(Finding("missing_chamber", f"{name} missing or very small", frac,
                               f">= {MIN_CHAMBER_FRACTION}"))
    return out


def findings_to_dicts(findings):
    return [asdict(f) for f in findings]
