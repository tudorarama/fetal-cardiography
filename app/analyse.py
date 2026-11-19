"""Analyses fetal four-chamber images from the command line.

Run from the project root with the four_chambers environment:
    four_chambers\\.venv\\Scripts\\python.exe -m app.analyse data/FOCUS/testing/images/022.png
    four_chambers\\.venv\\Scripts\\python.exe -m app.analyse img1.png img2.png --out results/demo
Prints the measurements and flags of every image (JSON with --json); with --out it
also saves <name>.png (overlay + heart close-up) and <name>.json in that folder.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # also runs as "python app/analyse.py"

import cv2
import numpy as np

from app.pipeline import Pipeline, render, render_zoom


def report(name, a):
    lines = [f"{name}: {a.status.upper()}"]
    if a.doppler:
        lines.append("  colour Doppler image: not analysed (the networks saw only B-mode)")
        return "\n".join(lines)
    lines.append(f"  CTR: {a.ctr:.3f}" if a.ctr is not None else f"  CTR: not measured ({a.ctr_reasons})")
    if not a.heart_ok:
        lines.append("  chambers: heart not found clearly")
    elif not a.chambers_ok:
        q = a.chamber_quality
        lines.append(f"  chambers: implausible segmentation (coverage {q['coverage']:.2f}, "
                     f"solidity {q['solidity']:.2f}, layout {q['layout_ok']}), not measured")
    else:
        ch = a.chambers
        fr = ", ".join(f"{c} {v:.0%}" for c, v in ch["fractions"].items())
        lines.append(f"  LV/RV {ch['lv_rv']:.2f}   LA/RA {ch['la_ra']:.2f}   ({fr})")
    for f in a.findings:
        lines.append(f"  FLAG {f.code}: {f.message} ({f.value:.2f}; flagged when {f.limit})")
    lines.append("  time: " + ", ".join(f"{k} {v:.1f} s" for k, v in a.seconds.items()))
    return "\n".join(lines)


def figure(a):
    full = render(a)
    zoom = render_zoom(a)
    if zoom is None:
        return full
    h = full.shape[0]
    zoom = cv2.resize(zoom, (h, h), interpolation=cv2.INTER_AREA)
    return np.hstack([full, np.full((h, 8, 3), 255, np.uint8), zoom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="folder for the figures and JSON reports")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--no-tta", action="store_true", help="faster, slightly less accurate")
    args = ap.parse_args()

    pipe = Pipeline(device=args.device, tta=not args.no_tta)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        a = pipe.run(path)
        print(json.dumps({"image": str(path), **a.summary()}, default=float) if args.json else report(path.name, a))
        if args.out:
            cv2.imwrite(str(args.out / f"{path.stem}.png"), cv2.cvtColor(figure(a), cv2.COLOR_RGB2BGR))
            (args.out / f"{path.stem}.json").write_text(json.dumps(a.summary(), indent=2, default=float))


if __name__ == "__main__":
    main()
