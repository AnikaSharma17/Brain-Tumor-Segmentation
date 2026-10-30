# Brain Tumor Segmentation using U-Net

[![CI](https://github.com/AnikaSharma17/Brain-Tumor-Segmentation-Using-U-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/AnikaSharma17/Brain-Tumor-Segmentation-Using-U-Net/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements and compares four U-Net variants with pretrained encoder
backbones — **ResNet50**, **VGG16**, **MobileNetV2**, and **EfficientNetB0** — to
segment brain tumors from 2D MRI slices.

---

## Objective

- Segment brain-tumor regions from MRI slices.
- Compare the four pretrained encoder backbones inside a shared U-Net decoder.
- Evaluate with segmentation-appropriate metrics on a held-out test set.

---

## Dataset

- **Source**: Figshare brain-tumor MRI dataset — 2D grayscale slices with binary
  tumor masks. (Not BRATS.)
- **Size**: 3,064 image/mask pairs.
- **Structure**:
  - Images: `image_dataset/images/`
  - Masks:  `image_dataset/masks/`  (binary, 0/255)
- **Loaded resolution**: 128×128.
- **Preprocessing** (see `src/brain_tumor_segmentation/data.py`):
  - Images are decoded as 3-channel (the single grayscale channel is replicated so
    the ImageNet-pretrained encoders can be used) and kept in `[0, 255]`; the
    correct per-encoder `preprocess_input` is applied **inside each model**.
  - Masks are resized with **nearest-neighbour** interpolation and re-binarized to
    `{0, 1}` (bilinear resizing would corrupt the labels at tumor boundaries).
  - Images and masks are paired explicitly by filename.
- **Split**: disjoint train / validation / test (≈70 / 15 / 15) with a fixed seed.
  > Note: filenames do not expose patient IDs, so this is a **slice-level** split.
  > Slices from the same patient may land in different splits; use a patient-grouped
  > split if patient IDs become available.
- **Augmentation** (training set only): random horizontal/vertical flips, 90°
  rotations, and additive Gaussian noise (geometric transforms are applied
  identically to image and mask).

---

## Architecture

- **Base**: U-Net (encoder–decoder with skip connections).
- **Encoders**: ResNet50, VGG16, MobileNetV2, EfficientNetB0 (ImageNet weights).
- **Output**: single-channel sigmoid mask at 128×128.
- **Loss**: combined binary cross-entropy + Dice (handles the small-foreground
  class imbalance).

---

## Metrics

- Dice Coefficient, Precision, Recall, Binary IoU (thresholded at 0.5), and pixel
  accuracy. Dice and IoU are the primary segmentation metrics; pixel accuracy is
  reported for reference but is inflated by the large background class.

---

## Project structure

```
.
├── src/brain_tumor_segmentation/   # importable package
│   ├── config.py          # paths & hyperparameters (env-overridable)
│   ├── data.py            # pairing, disjoint split, augmentation
│   ├── model.py           # U-Net: from-scratch baseline + 4 encoder backbones
│   ├── losses.py          # BCE + Dice loss
│   ├── metrics.py         # Dice coefficient
│   ├── inference.py       # load model + predict on a single image
│   └── visualization.py   # prediction grids & training curves
├── scripts/
│   ├── train.py           # train + evaluate all models
│   ├── evaluate.py        # per-image Dice distribution + error analysis
│   ├── plot_results.py    # bar charts from results spreadsheet
│   ├── visualize_models.py# prediction grids from saved models
│   ├── predict.py         # CLI: predict a mask for one image
│   └── app.py             # optional Gradio demo
├── notebooks/             # inference_demo.ipynb (load model + predict + overlay)
├── tests/                 # pytest suite
├── .github/workflows/     # CI (ruff + pytest)
├── image_dataset/{images,masks}/
├── outputs/               # generated at runtime (git-ignored)
├── Dockerfile             # CPU training/inference image
├── pyproject.toml         # packaging + dependencies
├── requirements.txt
├── LICENSE
└── .env.example           # optional config overrides
```

---

## Architecture diagram

```mermaid
flowchart LR
    A["MRI slice<br/>128x128x3"] --> B["Encoder<br/>(scratch / ResNet50 /<br/>VGG16 / MobileNetV2 /<br/>EfficientNetB0)"]
    B -->|skip connections| D["Decoder<br/>(up-conv + concat)"]
    B --> C["Bottleneck"]
    C --> D
    D --> E["1x1 Conv + sigmoid"]
    E --> F["Binary tumor mask<br/>128x128x1"]
```

Per-encoder ImageNet preprocessing is applied **inside** each model, so a saved
model is self-contained for inference. The from-scratch U-Net is the baseline the
pretrained backbones are compared against.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e .              # installs the package + dependencies
```

- Python 3.9–3.11, TensorFlow 2.15 (Keras 2).
- **GPU note:** TF ≥ 2.11 has no native Windows GPU support, so training runs on
  **CPU** on native Windows. For GPU use WSL2, or Python 3.10 + TensorFlow 2.10.
- Optional: copy `.env.example` to `.env` to override paths or hyperparameters.

## Running

```bash
python scripts/train.py            # train + evaluate all 5 models (scratch + 4 encoders)
python scripts/evaluate.py         # per-image Dice distribution + error analysis
python scripts/plot_results.py     # bar charts from the results spreadsheet
python scripts/visualize_models.py # prediction grids from saved .h5 models
python scripts/predict.py --model outputs/models/unet_resnet_model.h5 \
    --image image_dataset/images/1.png --out prediction.png   # single-image inference
```

All artifacts are written under `outputs/` (git-ignored):
- `outputs/model_comparison_results.xlsx` — per-model test metrics.
- `outputs/models/` — trained `.h5` models.
- `outputs/predictions/` — prediction grids.
- `outputs/training_history/` — per-model training curves and CSV logs.
- `outputs/graphs/` — metric comparison bar charts.
- `outputs/evaluation/` — per-image Dice histograms and summary CSV.

> Note: No results are committed to the repo. Run `scripts/train.py` to generate honest,
> held-out results into `outputs/`.

## Results

Metrics below are from one run of `scripts/train.py` on the **held-out test set**
(459 slices) with the default config (128×128 input, batch size 8, 20 epochs, seed 42,
CPU). `scripts/train.py` writes them to `outputs/model_comparison_results.xlsx`; your
own run may differ slightly. Dice and Binary IoU are the primary segmentation metrics;
pixel accuracy is reported for reference but is inflated by the large background class.

| Model                  | Dice      | Binary IoU | Precision | Recall | Pixel Acc. |
| ---------------------- | :-------: | :--------: | :-------: | :----: | :--------: |
| U-Net + ResNet50       | **0.790** | **0.831**  |   0.839   | 0.766  |   0.993    |
| U-Net + EfficientNetB0 |   0.790   |   0.830    |   0.829   | 0.774  |   0.993    |
| U-Net + VGG16          |   0.777   |   0.821    |   0.837   | 0.743  |   0.993    |
| U-Net (scratch)        |   0.729   |   0.791    |   0.683   | 0.815  |   0.990    |
| U-Net + MobileNetV2    |   0.725   |   0.789    |   0.758   | 0.722  |   0.991    |

Both pretrained top performers (ResNet50 and EfficientNetB0) clearly beat the
from-scratch baseline. EfficientNetB0 requires its encoder BatchNorm layers to be
frozen during fine-tuning (handled automatically in `model.build_encoder`); without
this it converges on the training set but collapses to all-background at inference.

### Per-image Dice distribution

Per-image hard-Dice over the test set (`scripts/evaluate.py`, which also writes
per-model histograms and `per_image_dice_summary.csv` under `outputs/evaluation/`).
The median sits well above the mean for every model: typical slices score highly, with
a tail of hard cases (tiny or faint tumors, some scoring 0) pulling the mean down.

| Model                  | Mean  | Median | Std   | P10 (worst 10%) |
| ---------------------- | :---: | :----: | :---: | :-------------: |
| U-Net + ResNet50       | 0.761 | 0.866  | 0.262 |      0.336      |
| U-Net + EfficientNetB0 | 0.756 | 0.856  | 0.257 |    **0.382**    |
| U-Net + VGG16          | 0.732 | 0.851  | 0.280 |      0.235      |
| U-Net (scratch)        | 0.708 | 0.816  | 0.273 |      0.275      |
| U-Net + MobileNetV2    | 0.681 | 0.809  | 0.309 |      0.011      |

ResNet50 has the best average, while EfficientNetB0 is the most consistent (lowest
variance and the strongest worst-10% performance). MobileNetV2 is the least reliable:
a decent median but its worst slices are near-total misses.

### Model card (summary)

- **Task**: binary semantic segmentation of brain-tumor regions in 2D MRI slices.
- **Input**: single MRI slice, resized to 128×128 and replicated to 3 channels.
- **Output**: a 128×128 binary tumor mask (per-pixel sigmoid, thresholded at 0.5).
- **Training data**: Figshare brain-tumor MRI dataset (3,064 slices), slice-level
  train/val/test split with a fixed seed.
- **Intended use**: research and educational demonstration of encoder-backbone
  comparison for medical image segmentation.
- **Not for clinical use**: slice-level split (possible residual patient leakage),
  2D-only (no volumetric context), and no external/clinical validation. See
  [Limitations](#limitations).

## Tests & demo

```bash
pip install -e ".[dev]"   # pytest + ruff
pytest                    # unit tests (metrics, losses, split, model shapes)
ruff check .              # lint

pip install -e ".[demo]"  # gradio
python scripts/app.py --model outputs/models/unet_resnet_model.h5   # web demo
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on push/PR across Python 3.10–3.11.

## Docker

```bash
docker build -t brain-tumor-seg .
docker run --rm -v "$(pwd)/image_dataset:/app/image_dataset" \
    -v "$(pwd)/outputs:/app/outputs" brain-tumor-seg
```

---

## Limitations

- **Slice-level split, not patient-level.** Filenames expose no patient IDs, so
  slices from one patient may fall across splits; some residual leakage is possible.
- **CPU-only on native Windows** with TensorFlow 2.15 (see GPU note above).
- **2D only** — each slice is segmented independently; no 3D volumetric context.
- Pixel accuracy is reported but is inflated by the large background class; rely on
  Dice and IoU.

---

## Visualizations

Each prediction grid shows the input image, the ground-truth mask, and the
predicted mask.
