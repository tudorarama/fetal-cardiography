# Roadmap

Goal of the project: segment the four chambers of the fetal heart in 2D four-chamber
ultrasound images and detect congenital anomalies from the segmentation.

## Where the project stands (19 Nov 2026)

| Part | State | Main limitation |
|---|---|---|
| Heart and thorax (stage 1) | U-Net Dice 0.92 / 0.97 on FOCUS, classical baseline, CTR error 0.026 | finished: frozen |
| Four chambers (nnU-Net v2) | mean Dice 0.72 (LV 0.78, RV 0.73, LA 0.65, RA 0.71) | 113 training images from one machine; tested on only 10 images |
| Anomaly detection | textbook rules; on Heartbeat (91 CHD exams) sensitivity 0.19, best AUC 0.61 (CTR) | rules cannot learn from the labelled data; chambers plausible on only 35 % of Heartbeat images |
| Demo | `app/`: whole method on one image, browser page | |

On the 50 normal FOCUS test hearts, the demo flags 6 images (12 %), mostly through the
atrial ratio: the atria are the weak point of the segmentation.

## Plan

### Phase 1: annotation on Heartbeat (data is the bottleneck)
- `four_chambers/select_heartbeat.py`, one image per patient, patients in one set only:
  - 90 train + 10 validation: healthy (No_CHD) cross_val patients, half 2T and half 3T,
    pre-drawn by the model and corrected;
  - 60 test (30 CHD, 30 No_CHD): drawn by hand.
- 20 more FOCUS test images drawn by hand (10 -> 30).
- Done when: every annotation is finished; the Dice of each source has a 95 % confidence interval
  narrower than about +-0.05.

### Phase 2: retrain the four-chamber model
- Train on FOCUS + Heartbeat; 5-fold cross-validation and ensemble (nnU-Net's standard recipe).
- `evaluate.py`: Dice per source (FOCUS, Heartbeat No_CHD, Heartbeat CHD), with
  patient-level bootstrap confidence intervals.
- Target: Dice >= 0.80 on FOCUS, most Heartbeat segmentations plausible.

### Phase 3: anomaly detection
- 3a. Rerun the rules with the new model (interpretable baseline).
- 3b. Learned detector: per-exam measurements (CTR, chamber fractions and ratios, segmentation
  quality, gestational age) -> logistic regression, on Heartbeat's official folds, final check on
  the 2T test part. Compared with the rules (sensitivity, specificity, AUC).
- Patients annotated for training (htrain / hval) are left out of every CHD evaluation.
- 3c (if time allows). A CNN on the heart crop with the same folds: segmentation-based versus
  end-to-end detection.

### Phase 4: report and presentation
- Final results with confidence intervals, updated architecture diagram, limitations
  (the four-chamber view shows about half of CHD; the defect type is unknown; one non-expert
  annotator; Heartbeat is restricted).

## Not pursued, and why
- CAMUS (adult, apical views): no fetal thorax and no RV / RA labels.
- More FETAL_PLANES_DB training images: Heartbeat brings other machines and CHD cases at once.
- A U-Net versus nnU-Net comparison for the chambers: only at the end, if time allows.

## Data rules
Heartbeat may not be redistributed: its images, masks, annotations, image list, predictions and
the models trained on it stay local. Only code and aggregated numbers are versioned.
