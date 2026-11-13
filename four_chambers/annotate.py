"""Annotation tool for the four chambers (and optional landmarks) on FOCUS images.

Run from the four_chambers folder:
    python annotate.py            # continues where you left off
    python annotate.py --status   # progress only
    python annotate.py --start 7  # jump to the 7th image of the list
    python annotate.py --no-sam   # manual drawing only
    python annotate.py --extend   # append the remaining FOCUS train/val images to the list
    python annotate.py --prefill  # open images already outlined by the model (predict.py --prefill)
    python annotate.py --pick     # start on the image picker
    python annotate.py --add-planes  # append the FETAL_PLANES_DB images (select_planes.py)

Choosing an image: key l opens a grid of all 300 FOCUS images (filters: all,
train, val, test, to do, done; border colour = state). Click one to open it; an
image that is not in the list yet is added to it (that is how more images are
annotated). Test images are never pre-filled by the model.

Second source, FETAL_PLANES_DB (splits ptrain / pval / ptest, see select_planes.py):
the heart outline shown with g comes from the stage-1 U-Net, not from an expert.
Not every image is a four-chamber view: mark the others with x.

The list of images (annotations/selection.json) is fixed once, with a seed:
30 train, 10 val, 10 test, interleaved (3 train, 1 val, 1 test, ...), so that
any prefix of the list is still balanced if you stop early. --extend appends
the other train/val images (the test set stays the same).

Human in the loop (--prefill): an image without annotation opens with the
chambers predicted by the trained nnU-Net; check every contour, redraw the wrong
ones (c clears a structure) and save with n. "prefilled": true is kept in the
JSON so it is known which annotations started from the model.

Two drawing modes (key a switches):
  SAM mode (default when ../models/sam_vit_b_01ec64.pth exists, see sam_assist.py)
    left click      point INSIDE the structure      -> SAM proposes a mask
    left drag       box around the structure (more stable than a point)
    right click     point OUTSIDE (removes a leak)
    m               next SAM proposal (they are sorted by size)
    ENTER           accept the mask for the current structure
  manual mode
    left drag / clicks   trace the outline,   ENTER / right click  close it

Keys in both modes
    1 LV   2 RV   3 LA   4 RA
    optional landmarks, only for the cardiac axis (one is enough, spine preferred):
    5 spine (vertebral body)   6 aorta (descending aorta, dark circle in front of the spine)
    z     undo last point (or delete the current structure)
    c     clear the current structure
    e     contrast enhancement on/off       g   heart outline (ground truth) on/off
    n     save + next image (after the last one: the next train/val image not done yet)
    p     save + previous image
    l     choose an image (thumbnail grid of every FOCUS image)
    x     image not usable (chambers not visible) -> skipped, next. Asks for a second x
          if something is drawn; the drawing is kept, p then z brings it back
    q/ESC save + quit
Annotations are saved in original image coordinates in annotations/.
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from common import (clahe, crop, is_complete, labels_to_polygons, list_cases, load_annotation,
                    load_focus_masks, load_image, role, save_annotation, short_name, thorax_crop_box)
from config import (ALL_SPLITS, ANN_DIR, CHAMBERS as CLASSES, LANDMARKS as OPTIONAL,
                    PREDICTIONS_DIR)

PREFILL_DIR = PREDICTIONS_DIR / "prefill"

N_PER_SPLIT = {"train": 30, "val": 10, "test": 10}
PATTERN = ("train", "train", "train", "val", "test")
VIEW = 820          # displayed crop size in pixels
PANEL = 320         # help panel width
COLORS = {"LV": (60, 60, 255), "RV": (255, 140, 40), "LA": (40, 210, 255),
          "RA": (80, 230, 80), "spine": (255, 80, 255), "aorta": (200, 200, 255)}  # BGR
WIN = "Four-chamber annotation"
STRUCTURES = CLASSES + OPTIONAL  # keys 1-6; only the 4 chambers are required
DRAG = 8            # pixels of mouse movement that turn a click into a drag


def load_case(split, name):
    """Image and the FOCUS heart/thorax masks (only used to frame the view)."""
    return load_image(split, name), None, load_focus_masks(split, name)


def extend_selection(order, ann_dir):
    """Appends every train/val image not yet in the list (3 train, 1 val, ...); test is fixed."""
    have = {tuple(x) for x in order}
    rng = np.random.default_rng(1)
    rest = {s: [str(n) for n in rng.permutation(list_cases(s)) if (s, str(n)) not in have]
            for s in ("train", "val")}
    added = 0
    while any(rest.values()):
        for s in ("train", "train", "train", "val"):
            if rest[s]:
                order.append([s, rest[s].pop(0)])
                added += 1
    (Path(ann_dir) / "selection.json").write_text(json.dumps(order, indent=1))
    print(f"added {added} images to the list (now {len(order)})")
    return order


def add_planes(order, ann_dir):
    """Appends the FETAL_PLANES_DB candidates (3 ptrain, 1 pval, 1 ptest, ...)."""
    have = {tuple(x) for x in order}
    rest = {s: [n for n in list_cases(s) if (s, n) not in have] for s in ("ptrain", "pval", "ptest")}
    if not any(rest.values()):
        print("no FETAL_PLANES_DB images to add (run select_planes.py first?)")
        return order
    added = 0
    while any(rest.values()):
        for s in ("ptrain", "ptrain", "ptrain", "pval", "ptest"):
            if rest[s]:
                order.append([s, rest[s].pop(0)])
                added += 1
    (Path(ann_dir) / "selection.json").write_text(json.dumps(order, indent=1))
    print(f"added {added} FETAL_PLANES_DB images to the list (now {len(order)})")
    return order


def selection(ann_dir):
    path = Path(ann_dir) / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(0)
    picks = {s: list(rng.permutation(list_cases(s))[:n]) for s, n in N_PER_SPLIT.items()}
    order = []
    while any(picks.values()):
        for s in PATTERN:
            if picks[s]:
                order.append([s, str(picks[s].pop(0))])
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(order, indent=1))
    return order


def status(order, ann_dir):
    done = {s: 0 for s in ALL_SPLITS}
    skipped = 0
    for split, name in order:
        ann = load_annotation(split, name, ann_dir)
        if is_complete(ann):
            done[split] += 1
        elif ann and ann.get("skipped"):
            skipped += 1
    return done, skipped


class Annotator:
    def __init__(self, order, ann_dir, start, sam=None, prefill_dir=None, pick=False):
        self.order, self.ann_dir, self.i, self.sam = order, ann_dir, start, sam
        self.prefill_dir = prefill_dir
        self.picker = None
        self.enhance, self.guide, self.use_sam = True, True, sam is not None
        self.drawing, self.points, self.down = False, [], None
        self.confirm_skip = False
        self.status_cache = status(order, ann_dir)
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load()
        if pick:
            self.open_picker()

    # ----------------------------------------------------------------- state
    def view_image(self, i, enhance):
        split, name = self.order[i]
        img, _, gt = load_case(split, name)
        box = thorax_crop_box(gt["thorax"], margin=0.12)
        base = clahe(img, clip=3.0) if enhance else img
        return img, gt, box, crop(base, box, VIEW, cv2.INTER_LINEAR)

    def load(self):
        split, name = self.order[self.i]
        self.img, gt, self.box, self.view = self.view_image(self.i, self.enhance)
        self.scale = VIEW / self.box[2]
        self.heart_view = crop(gt["cardiac"], self.box, VIEW, cv2.INTER_NEAREST)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self.allowed = {"heart": cv2.dilate(self.heart_view, k).astype(bool),
                        "thorax": cv2.dilate(crop(gt["thorax"], self.box, VIEW, cv2.INTER_NEAREST),
                                             k).astype(bool)}
        ann = load_annotation(split, name, self.ann_dir)
        self.ann = ann or {"split": split, "name": name,
                           "image_size": [self.img.shape[1], self.img.shape[0]],
                           "skipped": False, "polygons": {}}
        # start from the model's prediction (never for test images: the test set is drawn by hand)
        if ann is None and self.prefill_dir is not None and role(split) != "test":
            p = Path(self.prefill_dir) / f"{split}_{name}.png"
            if p.exists():
                self.ann["polygons"] = labels_to_polygons(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
                self.ann["prefilled"] = True
        self.cls = self.next_missing() or CLASSES[0]
        self.reset_prompt()
        if self.sam is not None:
            self.message("SAM is reading the image...")
            self.sam.set_image((self.i, self.enhance), self.rgb(self.view))
            for j in (self.i + 1, self.i - 1):  # prepare the neighbours in the background
                if 0 <= j < len(self.order):
                    self.sam.prefetch((j, self.enhance), self.rgb(self.view_image(j, self.enhance)[3]))

    @staticmethod
    def rgb(gray):
        return np.ascontiguousarray(np.stack([gray] * 3, -1))

    def reset_prompt(self):
        self.points, self.sam_pts, self.sam_box = [], [], None
        self.proposals, self.k = [], 0

    def save(self):
        if self.ann["polygons"] or self.ann.get("skipped"):
            save_annotation(self.ann, self.ann_dir)
        self.status_cache = status(self.order, self.ann_dir)

    def next_missing(self):
        return next((c for c in CLASSES if len(self.ann["polygons"].get(c, [])) < 3), None)

    # ----------------------------------------------------------- image picker
    def open_picker(self):
        from picker import ImagePicker
        self.save()
        self.reset_prompt()
        self.picker = ImagePicker(VIEW + PANEL, VIEW, self.ann_dir, self.order, self.order[self.i])

    def next_todo(self):
        """Next train/val image (FOCUS order) without a finished annotation, or None.

        Used by n at the end of the list. Test images are not added automatically:
        they are chosen on purpose in the picker and drawn by hand.
        """
        for split in ("train", "val", "ptrain", "pval"):
            for name in list_cases(split):
                ann = load_annotation(split, name, self.ann_dir)
                if not (is_complete(ann) or (ann and ann.get("skipped"))):
                    return split, name
        return None

    def goto(self, split, name):
        """Opens an image; if it is not in the list yet it is appended (and the list saved)."""
        case = [split, name]
        if case not in self.order:
            self.order.append(case)
            (Path(self.ann_dir) / "selection.json").write_text(json.dumps(self.order, indent=1))
        self.i = self.order.index(case)
        self.picker = None
        self.load()
        self.status_cache = status(self.order, self.ann_dir)

    # ------------------------------------------------------------ coordinates
    def to_view(self, pts):
        x0, y0, _ = self.box
        return [((x - x0) * self.scale, (y - y0) * self.scale) for x, y in pts]

    def to_image(self, x, y):
        x0, y0, _ = self.box
        return [round(x / self.scale + x0, 1), round(y / self.scale + y0, 1)]

    # ------------------------------------------------------------------- SAM
    def others_mask(self):
        """Pixels already taken by the other structures (a new mask may not overlap them)."""
        m = np.zeros((VIEW, VIEW), np.uint8)
        for c, pts in self.ann["polygons"].items():
            if c != self.cls and len(pts) >= 3:
                cv2.fillPoly(m, [np.round(self.to_view(pts)).astype(np.int32)], 1)
        return m.astype(bool)

    def run_sam(self):
        if not self.sam_pts and self.sam_box is None:
            self.proposals = []
            return
        allowed = self.allowed["thorax" if self.cls in OPTIONAL else "heart"] & ~self.others_mask()
        pts = [(x, y) for x, y, _ in self.sam_pts]
        labels = [l for _, _, l in self.sam_pts]
        self.proposals = self.sam.proposals(pts, labels, self.sam_box, allowed)
        self.k = 0

    def accept_sam(self):
        from sam_assist import mask_to_polygon
        if self.proposals:
            poly = mask_to_polygon(self.proposals[self.k])
            if len(poly) >= 3:
                self.ann["polygons"][self.cls] = [self.to_image(x, y) for x, y in poly]
                self.ann["skipped"] = False
                self.cls = self.next_missing() or self.cls
        self.reset_prompt()

    # ---------------------------------------------------------------- events
    def on_mouse(self, event, x, y, flags, _):
        if self.picker is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                chosen = self.picker.click(x, y)
                if chosen:
                    self.goto(*chosen)
            elif event == cv2.EVENT_MOUSEWHEEL:
                self.picker.turn(-1 if cv2.getMouseWheelDelta(flags) > 0 else 1)
            return
        if x >= VIEW or y >= VIEW:
            return
        if self.use_sam:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.down, self.drag_to = (x, y), (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.down:
                self.drag_to = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.down:
                (x0, y0), self.down = self.down, None
                if abs(x - x0) > DRAG or abs(y - y0) > DRAG:
                    self.sam_box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
                else:
                    self.sam_pts.append((x, y, 1))
                self.run_sam()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.sam_pts.append((x, y, 0))
                self.run_sam()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points.append(self.to_image(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            last = self.to_view([self.points[-1]])[0]
            if np.hypot(x - last[0], y - last[1]) > 5:
                self.points.append(self.to_image(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.close_polygon()

    def close_polygon(self):
        if len(self.points) >= 3:
            self.ann["polygons"][self.cls] = self.points
            self.ann["skipped"] = False
            self.cls = self.next_missing() or self.cls
        self.points = []

    def key(self, k, raw=None):
        """Returns False to quit. raw: full waitKeyEx code (arrow keys in the picker)."""
        if self.picker is not None:
            if self.picker.key(k if raw is None else raw) == "close":
                self.picker = None
            return True
        ch = chr(k) if 0 <= k < 256 else ""
        if ch == "x" and self.ann["polygons"] and not self.confirm_skip:
            self.confirm_skip = True  # never throw away a drawing on a single key press
            return True
        self.confirm_skip = False
        if ch and ch in "123456"[:len(STRUCTURES)]:
            self.cls = STRUCTURES[int(ch) - 1]
            self.reset_prompt()
        elif k in (13, 10):
            self.accept_sam() if self.use_sam else self.close_polygon()
        elif ch == "m" and self.proposals:
            self.k = (self.k + 1) % len(self.proposals)
        elif ch == "a" and self.sam is not None:
            self.use_sam = not self.use_sam
            self.reset_prompt()
        elif ch == "z":
            if self.ann.get("skipped"):  # undo "not usable": bring the drawing back
                self.ann["skipped"] = False
                self.ann["polygons"] = self.ann.pop("discarded", {}) or self.ann["polygons"]
                self.cls = self.next_missing() or self.cls
            elif self.use_sam and (self.sam_pts or self.sam_box):
                if self.sam_pts:
                    self.sam_pts.pop()
                else:
                    self.sam_box = None
                self.run_sam()
            elif self.points:
                self.points.pop()
            else:
                self.ann["polygons"].pop(self.cls, None)
        elif ch == "c":
            self.reset_prompt()
            self.ann["polygons"].pop(self.cls, None)
        elif ch == "e":
            self.enhance = not self.enhance
            self.load_keep()
        elif ch == "g":
            self.guide = not self.guide
        elif ch == "l":
            self.open_picker()
        elif ch in ("n", "p", "x"):
            if ch == "x":  # the drawing is kept as a backup, z restores it
                backup = self.ann["polygons"] or self.ann.get("discarded", {})
                self.ann = {**self.ann, "skipped": True, "polygons": {}, "discarded": backup}
            self.save()
            step = -1 if ch == "p" else 1
            if 0 <= self.i + step < len(self.order):
                self.i += step
                self.load()
            elif ch != "p":  # end of the list: continue with the next image to do, never quit
                nxt = self.next_todo()
                if nxt:
                    self.goto(*nxt)
                else:
                    self.open_picker()
        elif k == 27 or ch == "q":
            self.save()
            return False
        return True

    def load_keep(self):
        """Reloads the view (e.g. contrast changed) without losing the current annotation."""
        ann, cls = self.ann, self.cls
        self.load()
        self.ann, self.cls = ann, cls

    # ---------------------------------------------------------------- render
    def message(self, text):
        frame = np.zeros((VIEW, VIEW + PANEL, 3), np.uint8)
        cv2.putText(frame, text, (40, VIEW // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)

    def render(self):
        if self.picker is not None:
            return self.picker.render()
        view = cv2.cvtColor(self.view, cv2.COLOR_GRAY2BGR)
        if self.guide:
            cnt, _ = cv2.findContours(self.heart_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(view, cnt, -1, (200, 200, 200), 1)
        fill = view.copy()
        for c, pts in self.ann["polygons"].items():
            p = np.round(self.to_view(pts)).astype(np.int32)
            cv2.fillPoly(fill, [p], COLORS[c])
            cv2.polylines(view, [p], True, COLORS[c], 2)
        if self.proposals:
            m = self.proposals[self.k]
            fill[m] = COLORS[self.cls]
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(view, cnt, -1, COLORS[self.cls], 1)
        view = cv2.addWeighted(fill, 0.3, view, 0.7, 0)
        for c, pts in self.ann["polygons"].items():
            m = np.mean(self.to_view(pts), axis=0).astype(int)
            cv2.putText(view, c, tuple(m), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.points:
            p = np.round(self.to_view(self.points)).astype(np.int32)
            cv2.polylines(view, [p], False, COLORS[self.cls], 2)
        for x, y, l in self.sam_pts:
            cv2.circle(view, (x, y), 5, (0, 255, 0) if l else (0, 0, 255), -1)
            cv2.circle(view, (x, y), 5, (0, 0, 0), 1)
        if self.sam_box is not None:
            cv2.rectangle(view, self.sam_box[:2], self.sam_box[2:], (255, 255, 255), 1)
        if self.use_sam and self.down:
            cv2.rectangle(view, self.down, self.drag_to, (200, 200, 200), 1)
        return np.hstack([view, self.panel()])

    def panel(self):
        p = np.full((VIEW, PANEL, 3), 30, np.uint8)
        split, name = self.order[self.i]
        done, skipped = self.status_cache
        grey = (200, 200, 200)
        mode = "SAM (a: manual)" if self.use_sam else ("manual (a: SAM)" if self.sam else "manual")
        lines = [(f"{self.i + 1}/{len(self.order)}: {short_name(split, name)}", (255, 255, 255)),
                 (f"FOCUS  train {done['train']} val {done['val']} test {done['test']}", grey),
                 (f"PLANES train {done['ptrain']} val {done['pval']} test {done['ptest']}", grey),
                 (f"skipped {skipped}", grey), (f"mode: {mode}", (0, 255, 255)), ("", None),
                 ("now drawing:", grey), (f"  {self.cls}", COLORS[self.cls])]
        if self.use_sam:
            lines.append((f"  proposal {self.k + 1}/{len(self.proposals)}" if self.proposals
                          else "  (no proposal yet)", grey))
        lines.append(("", None))
        for i, c in enumerate(STRUCTURES, 1):
            ok = len(self.ann["polygons"].get(c, [])) >= 3
            opt = " (optional)" if c in OPTIONAL else ""
            lines.append((f"{i} {c:6s} {'OK' if ok else '--'}{opt}", COLORS[c]))
        lines.append(("", None))
        if self.use_sam:
            lines += [("click: point inside", grey), ("drag: box around", grey),
                      ("right click: point outside", grey), ("m: other proposal", grey),
                      ("ENTER: accept", grey)]
        else:
            lines += [("drag / click: draw", grey), ("ENTER / right click: close", grey)]
        lines += [("z undo   c clear", grey), ("e contrast   g heart outline", grey),
                  ("n next   p previous", grey), ("l choose image", (0, 255, 255)),
                  ("x not usable   q quit", grey)]
        if self.ann.get("prefilled"):
            lines.append(("PREFILLED BY THE MODEL:", (0, 200, 255)))
            lines.append(("check / correct every chamber", (0, 200, 255)))
        if self.ann.get("skipped"):
            lines.append(("MARKED NOT USABLE", (0, 0, 255)))
            lines.append(("z = undo (restores drawing)", (0, 0, 255)))
        if self.confirm_skip:
            lines.append(("x again = NOT USABLE,", (0, 0, 255)))
            lines.append(("deletes the drawing!", (0, 0, 255)))
            lines.append(("any other key = cancel", (0, 0, 255)))
        for j, (t, col) in enumerate(lines):
            if col:
                cv2.putText(p, t, (10, 28 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        return p

    def run(self):
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKeyEx(20)  # full codes, so the arrow keys work in the picker
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                self.save()
                break
            if k != -1 and not self.key(k & 0xFF, k):
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, help="1-based position in the list")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-sam", action="store_true", help="manual drawing only")
    ap.add_argument("--ann-dir", default=str(ANN_DIR))
    ap.add_argument("--extend", action="store_true",
                    help="append the remaining FOCUS train/val images to the list")
    ap.add_argument("--prefill", nargs="?", const=str(PREFILL_DIR), default=None,
                    help="folder of predicted label maps used as a starting point")
    ap.add_argument("--pick", action="store_true", help="start on the image picker")
    ap.add_argument("--add-planes", action="store_true",
                    help="append the FETAL_PLANES_DB images of planes_split.json to the list")
    args = ap.parse_args()

    order = selection(args.ann_dir)
    if args.extend:
        order = extend_selection(order, args.ann_dir)
    if args.add_planes:
        order = add_planes(order, args.ann_dir)
    done, skipped = status(order, args.ann_dir)
    print(f"annotated: FOCUS train {done['train']}, val {done['val']}, test {done['test']} | "
          f"FETAL_PLANES_DB train {done['ptrain']}, val {done['pval']}, test {done['ptest']} "
          f"(list of {len(order)} images), skipped {skipped}")
    if args.status:
        return
    sam = None
    if not args.no_sam:
        from sam_assist import SamAssistant, available
        if available():
            print("loading SAM...")
            sam = SamAssistant()
        else:
            print("SAM not available (pip install segment-anything torchvision and put "
                  "sam_vit_b_01ec64.pth in ../models/): manual mode only")
    if args.start:
        start = args.start - 1
    else:
        start = next((i for i, (s, n) in enumerate(order)
                      if not (is_complete(a := load_annotation(s, n, args.ann_dir))
                              or (a and a.get("skipped")))), len(order) - 1)
    Annotator(order, args.ann_dir, start, sam, args.prefill, args.pick).run()
    done, skipped = status(order, args.ann_dir)
    print(f"saved. annotated: FOCUS train {done['train']}, val {done['val']}, test {done['test']} | "
          f"FETAL_PLANES_DB train {done['ptrain']}, val {done['pval']}, test {done['ptest']}, "
          f"skipped {skipped}")


if __name__ == "__main__":
    main()
