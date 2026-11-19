"""End-to-end demo: one fetal four-chamber image -> outlines, chambers, measurements, flags."""
import subprocess
import sys
from pathlib import Path

FOUR_CHAMBERS = Path(__file__).resolve().parent.parent / "four_chambers"


def use_four_chambers_python():
    """Re-runs the current script with four_chambers/.venv if started with another Python.

    The demo needs nnU-Net, MONAI and Gradio, which are installed only there, so
    "py -m app.demo" would otherwise fail with "No module named 'gradio'".
    """
    venv = FOUR_CHAMBERS / ".venv"
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if Path(sys.prefix).resolve() == venv.resolve() or not python.exists():
        return
    print(f"(using {python})", flush=True)
    sys.exit(subprocess.run([str(python), sys.argv[0], *sys.argv[1:]]).returncode)
