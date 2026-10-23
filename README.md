# Fetal Echocardiography

Segmentation of Cardiac Structures (the Four Chambers of the Fetal Heart) in 2D Ultrasound Images and Detection of Congenital Anomalies

## Project Description


## State of the Art

Analysis of similar articles / applications / projects from the last 5 years.

## Literature review

| No. | Author(s) / Year | Title of the article / project | Application / Domain | Technologies used | Methodology / Approach | Results | Limitations | Additional comments |
|-----|------------------|--------------------------------|----------------------|-------------------|------------------------|---------|-------------|---------------------|
| 1 | Zhang et al. / 2025 | [You Can Detect It: Fetal Biometric Estimation Using Ellipse Detection](https://papers.miccai.org/miccai-2025/paper/5236_paper.pdf) + [FOCUS dataset](https://doi.org/10.5281/zenodo.14597550) | Finding the heart and the chest on fetal four-chamber ultrasound images and computing the heart-to-chest ratio (CTR) | PyTorch, ResNet50, U-Net, TransUNet | They detect the heart and the chest as ellipses instead of drawing exact contours. They also publish the FOCUS dataset (300 images with labels) | Dice 93.2% for the heart and 94.9% for the chest. U-Net got 89.1% and 94.0% | Only 300 images, all from one source | This is the dataset I use in my project, so I can compare my U-Net with their results |
| 2 | Le et al. / 2026 | [Ultra-ECP: Ellipse-Constrained and Point-Robust Foundation Model Adaptation for Fetal Cardiac Ultrasound Segmentation](https://proceedings.mlr.press/v315/le26a.html) | Segmenting the fetal heart and chest on the four-chamber view | UltraSAM (a "Segment Anything" model for ultrasound), LoRA | They take a large pretrained model and fine-tune only a small part of it. The user clicks a point and the model segments the region | Dice 95.1% for the chest and 92.6% for the heart on FOCUS | Not fully automatic (needs a click); the code is not public | Best result I found on FOCUS, useful as a target for my methods |
| 3 | Liang et al. / 2024 | [Automatic segmentation of 15 critical anatomical labels and measurements of cardiac axis and cardiothoracic ratio in fetal four chambers using nnU-NetV2](https://doi.org/10.1186/s12911-024-02527-x) | Segmenting 15 parts of the fetal heart and measuring the heart axis and the CTR | nnU-NetV2 | They segment the image, then fit lines and ellipses on the masks to get the measurements. They compare the results with doctors | Average Dice 87.1%. The CTR was close to what experienced doctors measured | Private data from one hospital; the heart axis was less accurate than the doctors | Shows how to get a medical measurement from a segmentation mask, like I do in my project |
| 4 | Li et al. / 2025 | [DCD: A Semantic Segmentation Model for Fetal Ultrasound Four-Chamber View](https://arxiv.org/abs/2506.08534) | Multi-class semantic segmentation of 13 anatomical structures in the fetal heart four-chamber view (4CH) | PyTorch, DeepLabv3+, Xception, CBAM, Dense ASPP | Enhanced DeepLabv3+ architecture with Xception backbone, CBAM attention modules, and Dense ASPP for fine boundary capture; trained with a hybrid loss ($L_{\text{CE}} + L_{\text{Dice}}$) | High mIoU and Dice scores across all 13 anatomical classes, significantly outperforming standard U-Net and DeepLabv3+ | High computational complexity during training due to the 13 anatomical classes | Provides a strong architectural baseline for detailed multi-class chamber segmentation in the 4CH view |
| 5 | Arnaout et al. / 2021 | [An ensemble of neural networks provides expert-level prenatal detection of complex congenital heart disease](https://doi.org/10.1038/s41591-021-01342-5) | Detecting heart defects in babies before birth | Several CNNs combined | First the model finds the right ultrasound views, then it decides if the heart is normal or not. A separate model segments the heart for measurements | AUC 0.99, sensitivity 95%, specificity 96% | Tested on old (retrospective) data with very few sick cases; in a later test it gave many false alarms | Shows that segmentation can also help with finding heart defects |

- **Application / Domain** – domain in which the described solution is applied (e.g., face recognition, augmented reality).
- **Technologies used** – technologies / hardware used in the project (e.g., OpenCV, TensorFlow, ZED stereo cameras).
- **Methodology / Approach** – description of the method used in the project.
- **Results** – main results obtained (metrics used for evaluation).
- **Limitations** – limitations encountered in research or implementation.
- **Additional comments** – observations on relevance to the current project; suggestions for improving the methodology.

## Proposed Architecture


### Components