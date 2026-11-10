# Four-chamber segmentation with nnU-Net v2

Segmentation of the four chambers of the fetal heart (LV, RV, LA, RA) in 2D
four-chamber ultrasound images of the FOCUS dataset, with
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), and measurements derived
from the masks for anomaly screening.

This folder is self-contained. It only reads the FOCUS images from
`../data/FOCUS` and the SAM weights from `../models/`.

## Why nnU-Net v2

- **It configures itself from the data.** It picks the preprocessing, network
  depth, patch size, batch size, augmentations and post-processing, so the
  method is a well-tested recipe and not a set of hand-tuned choices
  (Isensee et al., *Nature Methods* 2021).
- **It is still a strong baseline.** In the authors' 2024 re-evaluation
  ("nnU-Net Revisited", MICCAI 2024), CNN-based nnU-Net remains state of the
  art against newer Transformer and Mamba models.
- **It has been used on this view.** Liang et al. (2024, see the literature
  table) used nnU-Net v2 on the fetal four-chamber view, with 15 structures and
  an average Dice of 87%.

## Data and annotations

No public dataset with fetal chamber masks is available (FETUS 2026 could not
be accessed), so the chambers are annotated by hand on FOCUS images:

- **Image list** (`annotations/selection.json`): 30 train, 10 validation and 10
  test images, chosen with a fixed seed and interleaved. `--extend` adds the
  other 210 train/val images.
- **Per image**: polygons of LV, RV, LA and RA (required), plus optional
  landmarks (spine, else descending aorta) used only for the cardiac axis.
- **Annotation tool**: `annotate.py` (OpenCV window). Segment Anything (SAM
  ViT-B) proposes an outline from one click or a box. The annotator decides
  which structure it is, corrects it and accepts it.
- **Human in the loop**: after a first model is trained, `predict.py --prefill`
  segments the images that are not annotated yet, and `annotate.py --prefill`
  opens them pre-drawn. The annotator only checks and corrects. These files are
  marked `"prefilled": true`.

The test images are only ever drawn by hand and are never used for training.

## Pipeline

```
FOCUS image ──► square crop around the FOCUS heart outline (+20 %), 384 × 384
            ──► nnU-Net v2, 2D configuration: softmax over background + 4 chambers
            ──► probabilities resized back into the full image, argmax
            ──► masks ──► Dice / HD95, LV/RV and LA/RA area ratios, cardiac axis,
                          anomaly flags
```

| Step | File |
|---|---|
| Annotation (SAM-assisted) | `annotate.py`, `sam_assist.py` |
| Export to nnU-Net format (crop, labels, fold 0 = FOCUS train/val) | `prepare_dataset.py` |
| Plan, preprocess and train | `train.py`, `trainers.py` |
| Inference | `predict.py` |
| Metrics, measurements, anomaly rules | `evaluate.py`, `measurements.py` |

### Choices made for a GTX 1660 SUPER (6 GB) and why

| Problem | Measured | Solution |
|---|---|---|
| nnU-Net on the full 959 × 661 image (patch 768 × 1024) | 814 s per epoch, about 56 h for 250 epochs | crop around the heart: the chambers are all inside it, and FOCUS gives the heart outline for every image |
| Default schedule of 1000 epochs × 250 iterations, made for large 3D datasets | | `trainers.py`: E epochs × 50 iterations. Only the schedule changes; network, loss, augmentation and learning-rate policy stay nnU-Net's |
| Mixed precision (fp16) on GTX 16xx cards | training loss NaN from the first epoch, 27 s per epoch | full precision (fp32): loss normal and about 8 s per epoch, so 200 epochs take about 27 min |

## How to run

Run everything from this folder with its virtual environment (Python 3.14,
PyTorch 2.14.1 + CUDA 12.6, nnU-Net 2.8.1):

```
.venv\Scripts\python.exe annotate.py                  # annotate (continues where you stopped)
.venv\Scripts\python.exe annotate.py --status         # progress
.venv\Scripts\python.exe train.py --epochs 200        # export + plan + preprocess + train (fold 0)
.venv\Scripts\python.exe predict.py                   # segment the annotated test images
.venv\Scripts\python.exe evaluate.py                  # Dice, HD95, measurements -> results/
.venv\Scripts\python.exe predict.py --prefill         # pre-draw the images not annotated yet
.venv\Scripts\python.exe annotate.py --prefill        # check / correct them
```

Setting up the environment from scratch:

```
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Outputs

These folders are not versioned:

- `nnunet/raw`, `nnunet/preprocessed`: generated data
- `nnunet/results/.../fold_0/`: model, `progress.png` (training curve), log
- `nnunet/predictions/test/`: predicted label maps at full resolution
- `results/evaluation_test.csv`, `results/examples_test.png`

## Limitations

- **Few annotations.** There are tens of images, made by one non-expert
  annotator. The annotations should be checked by a clinician.
- **Small test set.** 10 images give a rough estimate only.
- **The crop uses the FOCUS heart outline.** For images outside FOCUS the heart
  must be found first, for example with the heart/thorax U-Net of the first
  project.
- **No pathological cases in FOCUS.** The anomaly rules (chamber ratios, axis,
  missing chamber) can only be shown to give no false alarms on normal hearts.
