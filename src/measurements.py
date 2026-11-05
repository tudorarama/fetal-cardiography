"""Clinical measurements computed from the heart and thorax ellipses.

An ellipse is (cx, cy, a, b, angle_deg) with a, b semi-axes in pixels, the
format used by FOCUS and returned by postprocess.fit_ellipse.
"""
import math


def ellipse_area(e):
    return math.pi * e[2] * e[3]


def ellipse_circumference(e):
    """Ramanujan's approximation of the ellipse perimeter."""
    a, b = e[2], e[3]
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def ctr_area(heart, thorax):
    """Cardiothoracic area ratio (normal range in the literature roughly 0.25-0.35)."""
    return ellipse_area(heart) / ellipse_area(thorax)


def ctr_circumference(heart, thorax):
    """Cardiothoracic circumference ratio (normal around 0.5)."""
    return ellipse_circumference(heart) / ellipse_circumference(thorax)


def ctr_from_masks(heart_mask, thorax_mask):
    """Area ratio computed directly from pixel counts (no ellipse fitting)."""
    return float(heart_mask.astype(bool).sum() / thorax_mask.astype(bool).sum())


def major_axis_angle(e):
    """Orientation of the ellipse's major axis in degrees, in [0, 180)."""
    _, _, a, b, angle = e
    return (angle if a >= b else angle + 90) % 180


def cardiac_axis(heart, thorax):
    """Approximate cardiac axis: angle between the heart's and the thorax's major axes, in [0, 90].

    Clinically the cardiac axis is measured between the interventricular septum
    and the sternum-spine line. With only ellipse annotations this is an
    approximation, and it is unreliable when the thorax is almost circular
    (a close to b), because its major axis direction is then arbitrary.
    """
    d = abs(major_axis_angle(heart) - major_axis_angle(thorax)) % 180
    return min(d, 180 - d)
