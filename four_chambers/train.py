"""Trains nnU-Net v2 (2D configuration) on the annotated four-chamber images.

Run from the four_chambers folder:
    python train.py --epochs 250              # full run
    python train.py --epochs 250 --continue   # resume after an interruption
    python train.py --epochs 5                # quick check that everything works

Steps (each printed before it runs):
  1. prepare_dataset.py: annotations -> nnunet/raw (regenerated every time, so new
     annotations are always included)
  2. nnUNetv2_plan_and_preprocess: nnU-Net analyses the data (image size, spacing,
     intensities) and chooses the network (depth, patch size, batch size) by itself;
     this "self-configuration" is the main idea of nnU-Net (Isensee et al., 2021)
  3. splits_final.json copied next to the preprocessed data: fold 0 = FOCUS train/val
  4. nnUNetv2_train <dataset> 2d 0 -tr nnUNetTrainerFetal_<E>ep on the GPU

nnU-Net's default is 1000 epochs x 250 iterations; trainers.py defines shorter
schedules (E epochs x 50 iterations), the only change to nnU-Net's recipe.
The model, the training curve (progress.png) and the log are in
nnunet/results/Dataset501_FetalHeart4CH/nnUNetTrainerFetal_<E>ep__nnUNetPlans__2d/fold_0/
"""
import config  # noqa: E402  (first: switch to four_chambers/.venv if needed)
config.ensure_venv()
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import CONFIGURATION, DATASET_ID, DATASET_NAME, NNUNET_DIR, setup_nnunet_env

EPOCH_VARIANTS = (1, 5, 50, 100, 150, 200, 300)
LAST_TRAINER = NNUNET_DIR / "last_trainer.txt"


def trainer_name(epochs):
    return f"nnUNetTrainerFetal_{epochs}ep"


def install_trainers():
    """Copies trainers.py into the nnunetv2 package, where nnU-Net looks trainers up by name."""
    import nnunetv2
    dst = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "fetal4ch"
    dst.mkdir(exist_ok=True)
    (dst / "__init__.py").touch()
    shutil.copy(Path(__file__).resolve().parent / "trainers.py", dst / "trainers.py")


def tool(name):
    """Path of an nnU-Net command installed in this virtual environment."""
    exe = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    if not exe.exists():
        raise SystemExit(f"{exe} not found: activate four_chambers/.venv (pip install nnunetv2)")
    return str(exe)


def run(cmd):
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=os.environ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200, choices=EPOCH_VARIANTS)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="resume the interrupted training (skips steps 1-3)")
    ap.add_argument("--gpu-memory", type=float, default=None,
                    help="GPU memory target for planning in GB (default: nnU-Net's 8 GB)")
    args = ap.parse_args()
    setup_nnunet_env()
    install_trainers()

    if not args.cont:
        from prepare_dataset import export
        raw = export()
        # the split changes when annotations are added; nnU-Net refuses to overwrite an
        # existing, different splits_final.json, so the old one (derived data) is removed
        dst = NNUNET_DIR / "preprocessed" / DATASET_NAME / "splits_final.json"
        dst.unlink(missing_ok=True)
        cmd = [tool("nnUNetv2_plan_and_preprocess"), "-d", str(DATASET_ID), "-c", CONFIGURATION,
               "--verify_dataset_integrity", "-np", "4"]
        if args.gpu_memory:
            cmd += ["-gpu_memory_target", str(args.gpu_memory)]
        run(cmd)
        shutil.copy(raw / "splits_final.json", dst)
        print(f"\nsplits: {dst}")

    trainer = trainer_name(args.epochs)
    LAST_TRAINER.write_text(trainer)
    cmd = [tool("nnUNetv2_train"), str(DATASET_ID), CONFIGURATION, args.fold, "-tr", trainer]
    if args.cont:
        cmd.append("--c")
    run(cmd)


if __name__ == "__main__":
    main()
