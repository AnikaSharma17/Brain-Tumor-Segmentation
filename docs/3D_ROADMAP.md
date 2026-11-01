# 3D Volumetric Brain-Tumor Segmentation — Roadmap

This document plans the extension of the project from **2D binary slice segmentation**
(the current TF/Keras code) to **3D multi-class volumetric segmentation** with a 3D
U-Net, as a parallel track that leaves the existing 2D project intact.

## 1. Why 3D, and what changes

The current project segments single 2D grayscale MRI slices with a binary mask on CPU.
Real brain-tumor segmentation is **volumetric and multi-modal**: each patient is a 3D
volume across four aligned MRI sequences (**T1, T1Gd/T1ce, T2, T2-FLAIR**), and the
clinically meaningful targets are three nested tumor sub-regions — **ET** (enhancing
tumor), **TC** (tumor core), **WT** (whole tumor).

Going 3D changes three things at once:

1. **Compute** — a 3D U-Net uses far more memory/compute than 2D. A GPU is required.
2. **Framework** — the de-facto standard for 3D medical imaging is **PyTorch + MONAI**
   (NIfTI I/O, medical transforms, patch sampling, sliding-window inference, 3D models,
   Dice/Hausdorff metrics). TensorFlow is not used for this track.
3. **Data** — multi-modal NIfTI volumes with multi-class labels, not grayscale PNGs.

## 2. Target hardware (this machine)

- **NVIDIA RTX 5060 Laptop GPU, 8 GB VRAM, compute capability sm_120 (Blackwell).**
- **PyTorch install (critical):** Blackwell (RTX 50-series) needs **PyTorch ≥ 2.7 with
  CUDA 12.8** wheels; the default pip wheels fail with "no kernel image available".
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
  Verify: `python -c "import torch; print(torch.cuda.is_available(), 'sm_120' in torch.cuda.get_arch_list())"`
  (both should be `True`). Runs on **native Windows — no WSL2 needed**. TensorFlow has
  no sm_120 Windows support, which is a further reason this track is PyTorch-only.
- **8 GB VRAM budget:** whole volumes do not fit, so train on **patches**
  (96³ safe, 128³ tight), **batch size 1–2**, **mixed precision (AMP)**, and use
  **sliding-window inference** to predict full volumes. A 3D U-Net or MONAI SegResNet
  fits comfortably; SwinUNETR is possible at 96³ + AMP but tighter.

## 3. Dataset

| Option | What | Access | Use |
| --- | --- | --- | --- |
| **MSD Task01_BrainTumour** | Cleaned BraTS subset, 750 4D volumes (484 train + 266 test), 4 modalities, NIfTI `.nii.gz` (~7 GB) | **Open**, auto-downloads via MONAI `DecathlonDataset(..., download=True)` | ✅ Start here |
| **BraTS 2024 / 2025** | Full challenge set, richer labels & cases, sequestered test set | Synapse registration + Data Use Agreement | Scale up later |

## 4. Repo layout for the 3D track

Keep the 2D project untouched; add a parallel package and env:

```
src/bts3d/            # 3D package (PyTorch + MONAI)
  __init__.py
  config.py           # paths, patch size, modalities, hyperparameters
  data.py             # MONAI transforms, DecathlonDataset, patient-level split
  model.py            # 3D UNet / SegResNet builders
  train.py            # AMP training loop, sliding-window validation
  evaluate.py         # per-region Dice + Hausdorff95
  inference.py        # predict a full NIfTI volume
scripts/sanity_3d.py  # Phase 0: verify GPU + load one volume
requirements-3d.txt   # monai, nibabel (torch installed from cu128 index)
data_3d/              # downloaded datasets (git-ignored)
```

## 5. Phased plan

**Phase 0 — Environment & data sanity** *(this is being scaffolded now)*
- New `.venv-3d`; install PyTorch cu128 + MONAI + nibabel.
- `scripts/sanity_3d.py`: verify CUDA + sm_120, then (with `--download`) pull MSD Task01,
  load one volume, print shapes, and save a slice montage.

**Phase 1 — 3D data pipeline**
- Per-modality z-score normalization on the brain, foreground crop, optional isotropic
  resampling.
- **Patient-level** train/val/test split (leakage-free — each volume is one patient,
  fixing the slice-level caveat of the 2D project).
- Patch sampling (balanced fg/bg crops) + 3D augmentation (flips, rotations, intensity).

**Phase 2 — Baseline 3D model**
- MONAI `UNet` or `SegResNet`: 4 input channels → 3 output channels (ET/TC/WT, multi-label).
- Dice + cross-entropy loss, AMP, sliding-window inference, checkpoint best val Dice.

**Phase 3 — Training & 3D evaluation**
- Metrics: **Dice per sub-region (ET/TC/WT)** and **Hausdorff95**, plus sensitivity.
- Per-case distribution + worst-case gallery (same philosophy as the 2D `evaluate.py`).

**Phase 4 — Push accuracy (optional)**
- nnU-Net (self-configuring, SOTA); UNETR / SwinUNETR transformers; deep supervision;
  3D TTA; ensembling; connected-component post-processing.

**Phase 5 — Package & present**
- Inference CLI (NIfTI in → mask + per-region Dice), 3D visualization, model card,
  README section, 2D-vs-3D comparison.

## 6. Advanced add-ons (to make it stand out)

- Multi-class sub-region segmentation (ET/TC/WT) vs the current binary mask.
- **nnU-Net** integration — reference-framework credibility.
- Transformer backbones (UNETR, SwinUNETR).
- Uncertainty estimation (MC-dropout / deep ensembles → confidence maps).
- Survival / outcome prediction head (BraTS ships clinical metadata) — multi-task.
- MLOps: experiment tracking (Weights & Biases / MLflow), DVC for multi-GB data.
- Deployment: ONNX export / MONAI Deploy, or a 3D Gradio demo.
- Federated learning (FeTS, the federated BraTS) for a research angle.

## 7. Risks

- **VRAM (8 GB)** — mitigated by patch training + AMP; keep patch/batch modest.
- **Blackwell setup** — must use cu128 PyTorch wheels (see §2).
- **Data size** — Task01 ≈ 7 GB; use MONAI `CacheDataset` (partial cache).
- **Framework ramp** — MONAI/PyTorch is a new stack vs the 2D Keras code.
- **Laptop thermals** — epochs slower than a desktop GPU; expect longer runs.

## 8. First concrete step

Prove the loop end to end on this GPU: `.venv-3d` + MONAI + MSD Task01 → a patch-based
3D U-Net reporting ET/TC/WT Dice. Then decide between scaling up (BraTS 2024/2025,
nnU-Net, transformers) and polishing (packaging, viewer, MLOps).

## Sources

- MONAI DecathlonDataset (Task01_BrainTumour auto-download): https://docs.monai.io/en/0.3.0/_modules/monai/apps/datasets.html
- Medical Segmentation Decathlon (AWS Open Data): https://registry.opendata.aws/msd/
- MONAI 3D brain-tumor segmentation tutorial: https://docs.wandb.ai/tutorials/monai_3d_segmentation
- MONAI 3D Segmentation – BraTS (patch training, sliding-window): https://oss-ai-ml-medical-segmentation.readthedocs.io/en/latest/MONAI%20Tutorials/3D%20Segmentation%20-%20BRATS%20Brain%20Tumors.html
- BraTS data access (The Cancer Imaging Archive): https://wiki.cancerimagingarchive.net/x/LojECQ
- PyTorch sm_120 / RTX 5060 (Blackwell) support: https://discuss.pytorch.org/t/pytorch-support-for-sm-120-nvidia-geforce-rtx-5060/220941
