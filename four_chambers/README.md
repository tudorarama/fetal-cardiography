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

### Second source: FETAL_PLANES_DB

FOCUS comes from a single machine. To have more images and other machines
(Voluson E6 / S10, Aloka), **485 images of FETAL_PLANES_DB** (Zenodo 3904280)
are added. They are chosen from its 1,718 "fetal thorax" images of 755
patients: one image per patient (the images of a patient are nearly identical)
and only where the heart was found well. They have no labels at all:

1. `../scripts/segment_planes.py` runs the heart/thorax U-Net of the first
   project on all of them (CPU, about 3 minutes) and saves the masks next to
   the images (`data/ZENODO/masks/`). These stand in for the FOCUS heart outline:
   they frame the annotation view and give the crop for nnU-Net.
2. `select_planes.py` picks candidates, one image per patient, split by patient:
   every official training patient (348 train `ptrain`, 87 validation `pval`)
   and 50 test images (`ptest`) from the official test patients
   (`planes_split.json`). Running it again with larger numbers only appends
   new patients, so started annotations keep their set.
3. `annotate.py --add-planes` appends them to the annotation list. Not all
   thorax images are four-chamber views: the others are marked with x.

`train` + `ptrain` are trained on, `val` + `pval` validate, `test` + `ptest`
are the test set, and `evaluate.py` reports the Dice per source as well.

## Pipeline

```
image ──► square crop around the heart outline (+20 %), 384 × 384
          (FOCUS: expert outline; FETAL_PLANES_DB: stage-1 U-Net)
            ──► nnU-Net v2, 2D configuration: softmax over background + 4 chambers
            ──► probabilities resized back into the full image, argmax
            ──► masks ──► Dice / HD95, LV/RV and LA/RA area ratios, cardiac axis,
                          anomaly flags
```

| Step | File |
|---|---|
| Annotation (SAM-assisted) | `annotate.py`, `sam_assist.py` |
| Heart outline and choice of the FETAL_PLANES_DB images | `../scripts/segment_planes.py`, `select_planes.py` |
| Export to nnU-Net format (crop, labels, fold 0 = train/val of both sources) | `prepare_dataset.py` |
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
.venv\Scripts\python.exe train.py --epochs 200 --continue  # resume an interrupted training
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

## Results

Model trained on 113 annotated FOCUS images (103 train, 10 validation, 200
epochs), tested on the 10 FOCUS test images drawn by hand:

| | LV | RV | LA | RA | mean |
|---|---|---|---|---|---|
| Dice (mean) | 0.78 | 0.73 | 0.65 | 0.71 | **0.72** |
| HD95 (px) | 22.7 | 28.3 | 19.7 | 16.6 | |

Cardiac axis: mean error 4.4° against the annotation. On one test image an
atrium is missed (Dice 0 for LA, and for RA on another), so the atria are the
weak point. The first model, trained on 39 images, reached a mean Dice of 0.68.

## Trained model

The model (`checkpoint_final.pth`, 268 MB) is too large for Git; it is
attached to the GitHub Release
[four-chambers-v1](https://github.com/tudorarama/fetal-cardiography/releases/tag/four-chambers-v1).
Unzip `nnunet_four_chambers.zip` in this folder: it creates
`nnunet/results/Dataset501_FetalHeart4CH/...`. Then `predict.py` works
without training.

## Outputs

These folders are not versioned:

- `nnunet/raw`, `nnunet/preprocessed`: generated data
- `nnunet/results/.../fold_0/`: model, `progress.png` (training curve), log
- `nnunet/predictions/test/`: predicted label maps at full resolution
- `results/evaluation_test.csv`, `results/examples_test.png`

## Limitations

- **Few annotations.** There are tens of images, made by one non-expert
  annotator. The annotations should be checked by a clinician.
- **Small test set.** 10 FOCUS images (plus the FETAL_PLANES_DB test images
  once drawn) give a rough estimate only.
- **The heart outline of FETAL_PLANES_DB is predicted.** The stage-1 U-Net was
  trained on FOCUS only; where it misses part of the heart, the crop can cut a
  chamber. The thorax is often cut off in these images (zoomed in), so their
  CTR is not reliable.
- **No pathological cases in FOCUS.** The anomaly rules (chamber ratios, axis,
  missing chamber) can only be shown to give no false alarms on normal hearts.
