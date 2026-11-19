"""Interactive demo in the browser (Gradio), local only.

Run from the project root (any Python: it switches to four_chambers/.venv itself):
    four_chambers\\.venv\\Scripts\\python.exe -m app.demo            # http://127.0.0.1:7860
    four_chambers\\.venv\\Scripts\\python.exe -m app.demo --cpu      # without a GPU (slower)

Upload a four-chamber image or pick an example. The page shows the heart and
thorax outlines, the four chambers, the measurements and the anomaly flags, and
says which step refused the image when it is not usable. Heartbeat examples are
listed only when data/HEARTBEAT/ is present (restricted data: never share the page
publicly while they are loaded).
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # also runs as "python app/demo.py"

from app import use_four_chambers_python  # noqa: E402

use_four_chambers_python()

import gradio as gr
import pandas as pd

from app.pipeline import CHAMBER_COLOURS, CHAMBERS, ROOT, Pipeline, render, render_zoom, to_json

DATA = ROOT / "data"
EXPLAIN = {
    "cardiomegaly": "The heart takes more than 35 % of the thorax area (cardiomegaly).",
    "ctr_outside_reference": "The CTR is outside the range of the normal FOCUS training hearts.",
    "ventricle_disproportion": "One ventricle is much smaller than the other (LV/RV outside 0.5-2).",
    "atrium_disproportion": "One atrium is much smaller than the other (LA/RA outside 0.5-2).",
    "missing_chamber": "A chamber is missing or very small (< 3 % of the four chambers).",
    "abnormal_axis": "The cardiac axis is outside 25-65 degrees.",
}


def examples(n=4, seed=0):
    """A few images of each source present on this computer: (path, label)."""
    rng = random.Random(seed)
    out = []
    focus = sorted((DATA / "FOCUS" / "testing" / "images").glob("*.png"))
    out += [(p, "FOCUS test") for p in rng.sample(focus, min(n, len(focus)))]
    planes = sorted((DATA / "ZENODO" / "Images").glob("*Plane6*.png"))
    out += [(p, "FETAL_PLANES_DB") for p in rng.sample(planes, min(n, len(planes)))]
    for cls in ("No_CHD", "CHD"):
        hb = sorted((DATA / "HEARTBEAT" / "2T" / "test" / cls).glob("*/*.4C.01.png"))
        out += [(p, f"Heartbeat {cls} (restricted)") for p in rng.sample(hb, min(n // 2, len(hb)))]
    return out


def verdict_markdown(a):
    if a.doppler:
        return ("### Not analysed\nThis is a **colour Doppler** image; the networks were trained "
                "on grey-scale (B-mode) images only.")
    done = []
    if a.ctr is not None:
        done.append("CTR")
    if a.chambers_ok:
        done.append("four chambers")
    if not done:
        return "### Not usable\n" + _refusals(a)
    head = "### Screen positive: look again" if a.findings else "### No flag"
    body = [f"Checked: **{', '.join(done)}**."]
    body += [f"- **{f.code.replace('_', ' ')}**: {EXPLAIN.get(f.code, f.message)} "
             f"(value {f.value:.2f}; flagged when {f.limit})" for f in a.findings]
    refused = _refusals(a)
    if refused:
        body.append("\nNot checked:\n" + refused)
    body.append("\n*A flag is a reason for a specialist to look again, never a diagnosis.*")
    return head + "\n" + "\n".join(body)


def _refusals(a):
    out = []
    if a.ctr is None:
        out.append(f"- CTR: heart and thorax not usable together ({a.ctr_reasons}).")
    if not a.heart_ok:
        out.append("- Chambers: the heart was not found clearly.")
    elif not a.chambers_ok:
        q = a.chamber_quality
        why = []
        if q["coverage"] < 0.25:
            why.append(f"the chambers cover only {q['coverage']:.0%} of the heart")
        if q["solidity"] < 0.9:
            why.append("a chamber is split into pieces")
        if q["layout_ok"] is False:
            why.append("the chambers are in an impossible arrangement")
        out.append(f"- Chambers: segmentation not plausible ({'; '.join(why)}), so not measured.")
    return "\n".join(out)


def measurement_table(a):
    rows = [("CTR (heart / thorax area)", _fmt(a.ctr), "0.15 - 0.35")]
    ch = a.chambers
    if ch:
        rows += [("LV / RV area", _fmt(ch["lv_rv"]), "0.5 - 2"),
                 ("LA / RA area", _fmt(ch["la_ra"]), "0.5 - 2")]
        rows += [(f"{c} share of the chambers", f"{ch['fractions'][c]:.0%}", ">= 3 %") for c in CHAMBERS]
        q = a.chamber_quality
        rows += [("chambers / heart outline", f"{q['coverage']:.0%}", ">= 25 % (plausibility)"),
                 ("chambers in one piece", f"{q['solidity']:.2f}", ">= 0.90 (plausibility)")]
    return pd.DataFrame(rows, columns=["measurement", "value", "normal / required"])


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def legend():
    sw = " ".join(f"<span style='color:rgb{CHAMBER_COLOURS[c]}'>&#9632;</span> {c}" for c in CHAMBERS)
    return (f"{sw} &nbsp; <span style='color:rgb(255,140,0)'>&#9632;</span> heart outline "
            f"&nbsp; <span style='color:rgb(150,150,150)'>&#9632;</span> thorax outline")


def build(pipe):
    def analyse(path):
        if not path:
            return None, None, "Upload an image or pick an example.", None, ""
        a = pipe.run(path)
        times = ", ".join(f"{k} {v:.1f} s" for k, v in a.seconds.items())
        return render(a), render_zoom(a), verdict_markdown(a) + f"\n\n<small>{times}</small>", \
            measurement_table(a), to_json(a)

    ex = examples()
    with gr.Blocks(title="Fetal heart: four chambers and anomaly screening") as demo:
        gr.Markdown("# Fetal echocardiography: four chambers and anomaly screening\n"
                    "Four-chamber view → heart and thorax (U-Net) → four chambers (nnU-Net v2) "
                    "→ measurements → rule-based flags. Research prototype, not a medical device.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Four-chamber ultrasound image", height=360)
                run = gr.Button("Analyse", variant="primary")
                if ex:
                    gr.Examples([[str(p)] for p, _ in ex], inputs=image, label="Examples",
                                example_labels=[f"{lbl}: {p.stem}" for p, lbl in ex])
            with gr.Column(scale=2):
                with gr.Row():
                    full = gr.Image(label="Outlines and chambers", height=360)
                    zoom = gr.Image(label="Heart close-up", height=360)
                gr.HTML(legend())
                verdict = gr.Markdown()
                table = gr.Dataframe(label="Measurements", interactive=False,
                                     headers=["measurement", "value", "normal / required"])
                with gr.Accordion("Raw result (JSON)", open=False):
                    raw = gr.Code(language="json")
        outputs = [full, zoom, verdict, table, raw]
        run.click(analyse, inputs=image, outputs=outputs)
        image.change(analyse, inputs=image, outputs=outputs)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    pipe = Pipeline(device="cpu" if args.cpu else None)
    # the datasets may be junctions to other drives: allow their real location too
    allowed = sorted({str(p) for d in DATA.glob("*") if d.is_dir() for p in (d, d.resolve())})
    build(pipe).launch(server_name="127.0.0.1", server_port=args.port, share=False, allowed_paths=allowed)


if __name__ == "__main__":
    main()
