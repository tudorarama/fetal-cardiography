"""Image picker for the annotation tool: a grid of thumbnails of every FOCUS image.

Opened with the key l in annotate.py. Filters (buttons at the top): all, train,
val, test (both sources), FOCUS, PLANES (FETAL_PLANES_DB), to do, done. The border colour shows the state of each image:
    green   all 4 chambers drawn          orange  started (some chambers)
    red     marked as not usable          grey    not started
A white frame marks the image currently open. Images that are not yet in the
annotation list show "+ add": clicking them appends them to the list.

Keys: left / right arrow (or the mouse wheel, or the < > buttons) change page,
l or ESC goes back to the current image.
"""
import cv2
import numpy as np

from common import crop, is_complete, is_planes, list_cases, load_annotation, load_focus_masks, \
    load_image, role, short_name, thorax_crop_box
from config import ALL_SPLITS

FILTERS = ("all", "train", "val", "test", "FOCUS", "PLANES", "to do", "done")
COLS, ROWS = 6, 4
BAR = 60            # height of the top bar
LEGEND = 24         # height of the colour legend at the bottom
THUMB = 150         # thumbnail side
STATE_COLORS = {"done": (80, 200, 80), "started": (0, 165, 255), "skipped": (60, 60, 230),
                "new": (120, 120, 120)}  # BGR
KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)  # waitKeyEx codes (Windows, Linux)


def state(ann):
    if is_complete(ann):
        return "done"
    if ann and ann.get("skipped"):
        return "skipped"
    if ann and ann.get("polygons"):
        return "started"
    return "new"


class ImagePicker:
    def __init__(self, width, height, ann_dir, order, current):
        self.w, self.h, self.ann_dir = width, height, ann_dir
        self.position = {tuple(x): i for i, x in enumerate(order)}  # (split, name) -> list index
        self.current = tuple(current)
        self.all = [(s, n) for s in ALL_SPLITS for n in list_cases(s)]
        self.filter, self.page = "all", 0
        self.thumbs = {}
        self.tile_w = width // COLS
        self.tile_h = (height - BAR - LEGEND) // ROWS
        self.buttons, self.tiles = [], []
        self.refresh()
        self.jump_to_current()

    # ------------------------------------------------------------------ data
    def refresh(self):
        """Reads the state of every image once (not at every frame)."""
        self._states = {c: state(load_annotation(*c, self.ann_dir)) for c in self.all}

    def states(self):
        return self._states

    def items(self, states):
        f = self.filter
        if f in ("train", "val", "test"):
            return [c for c in self.all if role(c[0]) == f]
        if f in ("FOCUS", "PLANES"):
            return [c for c in self.all if is_planes(c[0]) == (f == "PLANES")]
        if f == "to do":
            return [c for c in self.all if states[c] in ("new", "started")]
        if f == "done":
            return [c for c in self.all if states[c] == "done"]
        return list(self.all)

    def n_pages(self, items):
        return max(1, -(-len(items) // (COLS * ROWS)))

    def jump_to_current(self):
        items = self.items(self.states())
        if self.current in items:
            self.page = items.index(self.current) // (COLS * ROWS)

    def thumb(self, case):
        if case not in self.thumbs:
            img = load_image(*case)
            box = thorax_crop_box(load_focus_masks(*case)["thorax"], margin=0.05)
            self.thumbs[case] = cv2.cvtColor(crop(img, box, THUMB, cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        return self.thumbs[case]

    # --------------------------------------------------------------- drawing
    def render(self):
        states = self.states()
        items = self.items(states)
        pages = self.n_pages(items)
        self.page = min(self.page, pages - 1)
        frame = np.full((self.h, self.w, 3), 25, np.uint8)

        # top bar: filter buttons, page buttons, help
        self.buttons = []
        x = 10
        for f in FILTERS:
            label = f
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            r = (x, 10, x + tw + 20, 40)
            active = f == self.filter
            cv2.rectangle(frame, r[:2], r[2:], (0, 200, 255) if active else (90, 90, 90), -1 if active else 1)
            cv2.putText(frame, label, (x + 10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (20, 20, 20) if active else (220, 220, 220), 1)
            self.buttons.append((r, ("filter", f)))
            x = r[2] + 8
        for label, action, bx in (("<", -1, x + 20), (">", 1, x + 150)):
            r = (bx, 10, bx + 34, 40)
            cv2.rectangle(frame, r[:2], r[2:], (90, 90, 90), 1)
            cv2.putText(frame, label, (bx + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            self.buttons.append((r, ("page", action)))
        cv2.putText(frame, f"page {self.page + 1}/{pages}", (x + 62, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1)
        n_done = sum(1 for v in states.values() if v == "done")
        cv2.putText(frame, f"{len(items)} images | done {n_done}/{len(self.all)} | click = open, "
                    "l/ESC = back", (x + 200, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        # grid of thumbnails
        self.tiles = []
        start = self.page * COLS * ROWS
        for k, case in enumerate(items[start:start + COLS * ROWS]):
            r, c = divmod(k, COLS)
            x0, y0 = c * self.tile_w, BAR + r * self.tile_h
            tx, ty = x0 + (self.tile_w - THUMB) // 2, y0 + 6
            frame[ty:ty + THUMB, tx:tx + THUMB] = self.thumb(case)
            st = states[case]
            cv2.rectangle(frame, (tx - 3, ty - 3), (tx + THUMB + 2, ty + THUMB + 2), STATE_COLORS[st], 3)
            if case == self.current:
                cv2.rectangle(frame, (tx - 7, ty - 7), (tx + THUMB + 6, ty + THUMB + 6), (255, 255, 255), 2)
            pos = self.position.get(case)
            tag = f"#{pos + 1}" if pos is not None else "+ add"
            cv2.putText(frame, short_name(*case), (tx, ty + THUMB + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (230, 230, 230), 1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (tx, ty), (tx + tw + 6, ty + th + 8), (0, 0, 0), -1)  # tag on the thumb
            cv2.putText(frame, tag, (tx + 3, ty + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0) if pos is None else (200, 200, 200), 1)
            self.tiles.append(((x0, y0, x0 + self.tile_w, y0 + self.tile_h), case))

        # legend
        lx = 10
        for name, label in (("done", "done"), ("started", "started"), ("skipped", "not usable"),
                            ("new", "not started")):
            cv2.rectangle(frame, (lx, self.h - 18), (lx + 12, self.h - 6), STATE_COLORS[name], -1)
            cv2.putText(frame, label, (lx + 18, self.h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1)
            lx += 130
        return frame

    # ---------------------------------------------------------------- events
    def click(self, x, y):
        """Returns (split, name) when an image was chosen, else None."""
        for (x0, y0, x1, y1), (kind, value) in self.buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if kind == "filter":
                    self.filter, self.page = value, 0
                    self.jump_to_current()
                else:
                    self.turn(value)
                return None
        for (x0, y0, x1, y1), case in self.tiles:
            if x0 <= x < x1 and y0 <= y < y1:
                return case
        return None

    def turn(self, step):
        pages = self.n_pages(self.items(self.states()))
        self.page = (self.page + step) % pages

    def key(self, raw):
        """Returns 'close' to go back, else None."""
        if raw in KEY_LEFT:
            self.turn(-1)
        elif raw in KEY_RIGHT:
            self.turn(1)
        elif raw & 0xFF in (27, ord("l")):
            return "close"
        return None
