"""Builds docs/architecture.drawio: the pipeline diagram with real example images.

Run from the project root (needs results/unet_pre.pth and results/classical_params.json):
    python scripts/make_architecture_diagram.py

The images are embedded in the .drawio file (base64), so it opens on its own in
draw.io / diagrams.net. Blocks with images:
    1 dataset           three FOCUS test images as they are
    3 preprocessing     the same images after median -> bilateral -> CLAHE
    4a / 4b             raw output of each method, before post-processing
    5 post-processing   fitted ellipses (U-Net), heart red / thorax green
    7 measurements      ellipse areas and major axes used for CTR and cardiac axis
Right-hand track (four chambers, four_chambers/ project, same image 018):
    4c annotation       hand-drawn LV / RV / LA / RA (SAM-assisted) + aorta landmark
    4d region of interest  384 x 384 crop around the FOCUS heart outline
    4e nnU-Net v2       predicted chambers in the crop
    7b measurements     chamber masks, septum line and antero-posterior line
Needs four_chambers/annotations/test_018.json and four_chambers/nnunet/predictions/test/
(run four_chambers/predict.py first).
"""
import base64
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.classical import load_params, segment
from src.data import load_case
from src.measurements import ctr_area, major_axis_angle
from src.postprocess import refine
from src.preprocess import preprocess
from src.unet import load_model, prepare, unletterbox

CASES = ("050", "018", "045")          # test images where both methods work well
DETAIL = "018"                         # image used for blocks 4a, 4b, 7
OUT = ROOT / "docs" / "architecture.drawio"
FC = ROOT / "four_chambers"
CHAMBER_COLORS = {1: (60, 60, 230), 2: (255, 120, 40), 3: (30, 200, 250), 4: (60, 200, 60)}  # LV RV LA RA
RED, GREEN = (40, 40, 230), (60, 200, 60)   # BGR: heart, thorax

# block colours (fill, stroke), same palette as the first diagram
GREY = ("#f5f5f5", "#666666")
BLUE = ("#dae8fc", "#6c8ebf")
GREEN_B = ("#d5e8d4", "#82b366")
PURPLE = ("#e1d5e7", "#9673a6")
YELLOW = ("#fff2cc", "#d6b656")
RED_B = ("#f8cecc", "#b85450")
ORANGE = ("#ffe6cc", "#d79b00")


# ------------------------------------------------------------------ images

def jpg(img, width):
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode(), img.shape[1], img.shape[0]


def bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def thorax_crop(img, thorax_mask, margin=0.12):
    ys, xs = np.nonzero(thorax_mask)
    side = int(max(np.ptp(xs), np.ptp(ys)) * (1 + 2 * margin))
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    out = np.zeros((side, side) + img.shape[2:], img.dtype)  # always square, black padding
    h, w = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def fill(vis, mask, color, alpha=0.45):
    m = mask.astype(bool)
    vis[m] = ((1 - alpha) * vis[m] + alpha * np.array(color)).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnt, -1, color, 2)
    return vis


def draw_ellipse(vis, e, color, thickness=3):
    cx, cy, a, b, ang = e
    cv2.ellipse(vis, (round(cx), round(cy)), (round(a), round(b)), ang, 0, 360, color, thickness)
    return vis


def draw_major_axis(vis, e, color):
    cx, cy, a, b, _ = e
    t = np.radians(major_axis_angle(e))
    r = max(a, b)
    p0 = (round(cx - r * np.cos(t)), round(cy - r * np.sin(t)))
    p1 = (round(cx + r * np.cos(t)), round(cy + r * np.sin(t)))
    cv2.line(vis, p0, p1, color, 2, cv2.LINE_AA)
    return vis


def make_images():
    model, use_pre = load_model(ROOT / "results" / "unet_pre.pth")
    params = load_params(ROOT / "results" / "classical_params.json")
    out = {"input": [], "pre": [], "post": []}

    for name in CASES:
        img, _, gt = load_case("test", name)
        pre = preprocess(img)
        out["input"].append(jpg(bgr(img), 300))
        out["pre"].append(jpg(bgr(pre), 300))

        x, side = prepare(img, use_pre)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
        raw = {k: (unletterbox(prob[i], img.shape, side) > 0.5).astype(np.uint8)
               for i, k in enumerate(("cardiac", "thorax"))}
        ell = {k: refine(m)[1] for k, m in raw.items()}
        vis = draw_ellipse(draw_ellipse(bgr(img), ell["thorax"], GREEN), ell["cardiac"], RED)
        out["post"].append((jpg(thorax_crop(vis, gt["thorax"]), 220),
                            ctr_area(ell["cardiac"], ell["thorax"])))

        if name == DETAIL:
            # 4a classical: raw thorax (radial search) and raw heart blob (Chan-Vese)
            r = segment(pre, params, keep_debug=True)
            d = r.debug
            up = lambda m: cv2.resize(m.astype(np.uint8), img.shape[::-1], interpolation=cv2.INTER_NEAREST)
            v = fill(fill(bgr(img), up(d["thorax_raw"]), GREEN, 0.25), up(d["heart_raw"]), RED)
            out["classical"] = jpg(thorax_crop(v, gt["thorax"]), 220)
            # 4b U-Net: thresholded probability maps
            v = fill(fill(bgr(img), raw["thorax"], GREEN, 0.25), raw["cardiac"], RED)
            out["unet"] = jpg(thorax_crop(v, gt["thorax"]), 220)
            # 7 measurements: filled ellipses + major axes
            v = bgr(img)
            for k, c in (("thorax", GREEN), ("cardiac", RED)):
                m = np.zeros(img.shape, np.uint8)
                cx, cy, a, b, ang = ell[k]
                cv2.ellipse(m, (round(cx), round(cy)), (round(a), round(b)), ang, 0, 360, 1, -1)
                v = fill(v, m, c, 0.3)
                v = draw_major_axis(v, ell[k], (255, 255, 255) if k == "cardiac" else (0, 255, 255))
            out["measure"] = jpg(thorax_crop(v, gt["thorax"]), 220)
            out["measure_ctr"] = ctr_area(ell["cardiac"], ell["thorax"])
    return out


def make_chamber_images(case=("test", DETAIL)):
    """Images of the four-chamber track (needs the four_chambers annotations and predictions)."""
    import importlib.util
    import json
    sys.path.append(str(FC))  # four_chambers modules import its own config.py
    spec = importlib.util.spec_from_file_location("fc_measurements", FC / "measurements.py")
    fcm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcm)

    split, name = case
    cid = f"{split}_{name}"
    img, _, gt = load_case(split, name)
    ann = json.loads((FC / "annotations" / f"{cid}.json").read_text())
    pred = cv2.imread(str(FC / "nnunet" / "predictions" / "test" / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
    roi_pred = cv2.imread(str(FC / "nnunet" / "predictions" / "test_roi" / f"{cid}.png"),
                          cv2.IMREAD_GRAYSCALE)
    roi_img = cv2.imread(str(FC / "nnunet" / "predictions" / "test_roi_input" / f"{cid}_0000.png"),
                         cv2.IMREAD_GRAYSCALE)
    out = {}

    def colorize(gray, lab):
        v = bgr(gray)
        for c, col in CHAMBER_COLORS.items():
            v = fill(v, (lab == c).astype(np.uint8), col, 0.45)
        return v

    # 4c annotation: polygons + landmark
    lab = np.zeros(img.shape, np.uint8)
    for c, k in enumerate(("LV", "RV", "LA", "RA"), 1):
        cv2.fillPoly(lab, [np.round(np.array(ann["polygons"][k])).astype(np.int32)], c)
    v = colorize(img, lab)
    landmark = None
    for k in ("spine", "aorta"):
        if len(ann["polygons"].get(k, [])) >= 3:
            pts = np.round(np.array(ann["polygons"][k])).astype(np.int32)
            cv2.polylines(v, [pts], True, (255, 80, 255), 2)
            landmark = tuple(np.mean(ann["polygons"][k], axis=0))
            break
    out["annotation"] = jpg(thorax_crop(v, gt["thorax"]), 220)

    # 4d region of interest, 4e nnU-Net prediction in it
    out["roi"] = jpg(bgr(roi_img), 200)
    out["roi_pred"] = jpg(colorize(roi_img, roi_pred), 200)

    # 7b measurements on the full-resolution prediction
    v = colorize(img, pred)
    m = fcm.chamber_measurements(pred, None, None)
    ys, xs = np.nonzero(gt["thorax"])
    center = (xs.mean(), ys.mean())
    axis = fcm.cardiac_axis(pred, landmark, center)
    sept = fcm.septum_direction(pred)
    if sept is not None:
        (ly, lx), (ry, rx) = [np.argwhere(pred == c).mean(axis=0) for c in (1, 2)]
        mx, my = (lx + rx) / 2, (ly + ry) / 2
        p0 = (round(mx - 110 * sept[0]), round(my - 110 * sept[1]))
        p1 = (round(mx + 110 * sept[0]), round(my + 110 * sept[1]))
        cv2.line(v, p0, p1, (255, 255, 255), 3, cv2.LINE_AA)
    if landmark is not None:
        lm = (round(landmark[0]), round(landmark[1]))
        d = np.array(center) - np.array(landmark)
        far = (round(landmark[0] + 2.2 * d[0]), round(landmark[1] + 2.2 * d[1]))
        cv2.line(v, lm, far, (255, 255, 0), 4, cv2.LINE_AA)
        cv2.circle(v, lm, 7, (255, 80, 255), -1)
    out["chamber_measure"] = jpg(thorax_crop(v, gt["thorax"]), 220)
    out["chamber_values"] = (m["lv_rv"], m["la_ra"], axis)
    return out


# ------------------------------------------------------------------- draw.io

class Diagram:
    def __init__(self):
        self.cells, self.n = [], 1

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, title, text, colors, container=False):
        i = self._id()
        label = f"<b style='font-size:15px'>{escape(title)}</b><br><span style='font-size:12px'>{text}</span>"
        style = (f"rounded=1;whiteSpace=wrap;html=1;arcSize=6;fillColor={colors[0]};"
                 f"strokeColor={colors[1]};strokeWidth=2;fontFamily=Helvetica;"
                 + ("verticalAlign=top;spacingTop=6;container=1;collapsible=0;" if container else ""))
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def image(self, parent, x, y, data, w=None):
        b64, iw, ih = data
        w = w or iw
        h = round(ih * w / iw)
        i = self._id()
        style = (f"shape=image;html=1;imageAspect=0;aspect=fixed;strokeColor=#999999;"
                 f"imageBorder=#999999;image=data:image/jpeg,{b64};")
        self.cells.append(f'<mxCell id="{i}" value="" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return h

    def text(self, parent, x, y, w, h, text, size=11, color="#333333", align="center", rotation=0):
        i = self._id()
        style = (f"text;html=1;align={align};verticalAlign=middle;whiteSpace=wrap;rotation={rotation};"
                 f"fontSize={size};fontColor={color};fontFamily=Helvetica;")
        self.cells.append(f'<mxCell id="{i}" value="{escape(text)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def edge(self, src, dst, label="", dashed=False, exit_=None, entry=None, points=()):
        i = self._id()
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;"
                 "strokeWidth=2;strokeColor=#333333;fontSize=11;labelBackgroundColor=#ffffff;"
                 + ("dashed=1;" if dashed else ""))
        if exit_:
            style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
        geo = (f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>'
               if pts else '<mxGeometry relative="1" as="geometry"/>')
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" edge="1" '
                          f'parent="1" source="{src}" target="{dst}">{geo}</mxCell>')

    def xml(self, w, h):
        body = "".join(self.cells)
        return (f'<mxfile host="app.diagrams.net"><diagram id="architecture" name="Architecture">'
                f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
                f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" '
                f'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}'
                f'</root></mxGraphModel></diagram></mxfile>')


def build(img):
    d = Diagram()
    W, X = 1900, 330            # page width, x of the central column
    RX, RW = 1360, 480          # right-hand track: four chambers
    CW = 640                    # central column width

    def row_of_images(parent, y, items, width, captions=None):
        gap = (CW - len(items) * width) / (len(items) + 1)
        for k, data in enumerate(items):
            x = round(gap + k * (width + gap))
            h = d.image(parent, x, y, data, width)
            if captions:
                d.text(parent, x - 10, y + h + 2, width + 20, 18, captions[k])

    # 1. dataset
    b1 = d.box(X, 20, CW, 240, "1. FOCUS dataset",
               "300 four-chamber ultrasound images (200 train / 50 val / 50 test) + heart &amp; thorax ellipse labels",
               GREY, container=True)
    row_of_images(b1, 62, img["input"], 190)

    # 2. loading
    b2 = d.box(X + 120, 300, CW - 240, 60, "2. Data loading & ground truth",
               "train / val / test split, ellipses → masks", GREY)

    # 3. preprocessing
    b3 = d.box(X, 400, CW, 240, "3. Preprocessing",
               "median filter (speckle) → bilateral filter (edges kept) → CLAHE (local contrast)",
               BLUE, container=True)
    row_of_images(b3, 62, img["pre"], 190)

    # 4a / 4b
    b4a = d.box(60, 690, 520, 330, "4a. Classical segmentation",
                "thorax: radial search of the wall + Otsu, robust ellipse fit<br>"
                "heart: hierarchical Chan-Vese inside the thorax", GREEN_B, container=True)
    d.image(b4a, 150, 68, img["classical"], 220)
    b4b = d.box(720, 690, 520, 330, "4b. Deep learning segmentation",
                "U-Net (MONAI) trained on FOCUS train, 2 sigmoid outputs (heart, thorax)",
                PURPLE, container=True)
    d.image(b4b, 150, 68, img["unet"], 220)
    d.text(b4a, 20, 294, 480, 22, "raw output: thorax region (green) and dark heart blob (red)")
    d.text(b4b, 20, 294, 480, 22, "raw output: probability maps thresholded at 0.5")

    # 5. post-processing
    b5 = d.box(X, 1070, CW, 300, "5. Post-processing",
               "largest component → hole filling → ellipse fitting (heart red, thorax green)",
               BLUE, container=True)
    row_of_images(b5, 62, [p[0] for p in img["post"]], 170,
                  captions=[f"CTR = {p[1]:.2f}" for p in img["post"]])

    # 6 evaluation, 7 measurements
    b6 = d.box(60, 1420, 520, 110, "6. Segmentation evaluation",
               "Dice and HD95 against the ground-truth masks<br>(test set used once, after tuning on val)",
               YELLOW)
    b7 = d.box(720, 1420, 520, 330, "7. Measurements",
               "cardiothoracic ratio CTR = area(heart) / area(thorax)<br>"
               "cardiac axis ≈ angle between the major axes", YELLOW, container=True)
    d.image(b7, 150, 68, img["measure"], 220)
    d.text(b7, 20, 296, 480, 22, f"example: CTR = {img['measure_ctr']:.2f} (normal range ≈ 0.25–0.35)")

    # 8 anomaly, 9 results
    b8 = d.box(720, 1800, 520, 110, "8. Anomaly detection (rules, screening only)",
               "CTR &gt; 0.35 → cardiomegaly &nbsp;|&nbsp; LV/RV or LA/RA outside 0.5–2 → disproportion<br>"
               "chamber &lt; 3 % of the heart → missing chamber &nbsp;|&nbsp; axis outside 25–65° → abnormal axis",
               RED_B)
    b9 = d.box(X, 1960, CW, 80, "9. Results & comparison",
               "heart/thorax: classical vs U-Net vs fixed-circle baseline (Wilcoxon) · chambers: Dice per chamber",
               ORANGE)

    # four-chamber track (four_chambers/ project)
    ch = img["chambers"]
    c1 = d.box(RX, 400, RW, 330, "4c. Chamber annotation (own)",
               "LV, RV, LA, RA drawn on FOCUS images; SAM proposes the outline from a click,<br>"
               "the annotator corrects it · optional landmark: spine / descending aorta",
               PURPLE, container=True)
    d.image(c1, 130, 80, ch["annotation"], 220)
    c2 = d.box(RX, 780, RW, 260, "4d. Region of interest",
               "square around the FOCUS heart outline (+20 %), resized to 384 × 384", BLUE, container=True)
    d.image(c2, 140, 52, ch["roi"], 200)
    c3 = d.box(RX, 1070, RW, 300, "4e. nnU-Net v2 (2D, self-configuring)",
               "trained from scratch on our annotations (fold 0 = FOCUS train / val)<br>"
               "fp32 on a GTX 1660; probabilities resized back into the full image", PURPLE, container=True)
    d.image(c3, 140, 80, ch["roi_pred"], 200)
    lv_rv, la_ra, axis = ch["chamber_values"]
    c5 = d.box(RX, 1420, RW, 340, "7b. Chamber measurements",
               "LV/RV and LA/RA area ratios · cardiac axis = septum (white) vs<br>"
               "spine/aorta → thorax centre line (cyan)", YELLOW, container=True)
    d.image(c5, 130, 80, ch["chamber_measure"], 220)
    d.text(c5, 20, 306, 440, 22, f"example: LV/RV = {lv_rv:.2f}, LA/RA = {la_ra:.2f}"
           + (f", axis = {axis:.0f}°" if axis is not None else ""))
    c4 = d.box(RX, 1960, RW, 80, "6b. Chamber evaluation",
               "Dice and HD95 per chamber against the hand annotation of the test images", YELLOW)

    # flow
    d.edge(b1, b2)
    d.edge(b2, b3)
    d.edge(b3, b4a, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b3, b4b, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b4a, b5, "predicted mask", exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b4b, b5, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b5, b6, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b5, b7, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(b7, b8)
    d.edge(b6, b9, exit_=(0.5, 1), entry=(0, 0.5), points=[(320, 1970)])
    d.edge(b8, b9, exit_=(0.3, 1), entry=(0.85, 0))
    # four-chamber track
    d.edge(b1, c1, "images", exit_=(1, 0.75), entry=(0.85, 0), points=[(1768, 200)])
    d.edge(c1, c2)
    d.edge(c2, c3)
    d.edge(c3, c5)
    d.edge(c5, b8, exit_=(0, 0.85), entry=(1, 0.5), points=[(1300, 1709), (1300, 1855)])
    d.edge(c3, c4, exit_=(1, 0.5), entry=(1, 0.5),
           points=[(1880, 1220), (1880, 2000)])
    d.edge(c4, b9, exit_=(0, 0.5), entry=(1, 0.5))
    # ground truth paths
    d.edge(b2, b6, dashed=True, exit_=(0, 0.5), entry=(0, 0.5),
           points=[(30, 330), (30, 1475)])
    d.edge(b2, b4b, "training masks", dashed=True, exit_=(1, 0.5), entry=(1, 0.5),
           points=[(1270, 330), (1270, 840)])
    # legend
    leg = d.box(1010, 20, 300, 120, "Legend", "<font color='#e62828'>■</font> heart (cardiac) "
                "&nbsp; <font color='#3cc83c'>■</font> thorax<br>"
                "chambers: <font color='#e63c3c'>■</font> LV <font color='#2878ff'>■</font> RV "
                "<font color='#fac81e'>■</font> LA <font color='#3cc83c'>■</font> RA<br>"
                "── data flow &nbsp; - - - ground truth", ("#ffffff", "#999999"))
    d.text("1", RX, 360, RW, 30, "<b>Four-chamber segmentation track</b>", 14, "#9673a6")
    d.text("1", 75, 612, 250, 30, "<b>Heart &amp; thorax track (CTR)</b>", 14, "#555555", align="left")
    # labels of the two long side edges, placed beside the line (not on it, not at the page edge)
    d.text("1", 40, 306, 160, 20, "ground-truth masks", 11, "#333333", align="left")
    d.text("1", 1820, 1590, 140, 20, "predicted chambers", 11, "#333333", rotation=90)
    return d.xml(W, 2070)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    images = make_images()
    images["chambers"] = make_chamber_images()
    OUT.write_text(build(images), encoding="utf-8")
    print(f"saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
