"""Central configuration for the 3D track (PyTorch + MONAI).

Mirrors the 2D package's config style: env-overridable defaults keep machine-specific
paths and hyperparameters out of the code. See docs/3D_ROADMAP.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# …/src/bts3d/config.py -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str | Path) -> Path:
    value = os.getenv(env)
    return Path(value) if value else Path(default)


def _int(env: str, default: int) -> int:
    return int(os.getenv(env, default))


def _float(env: str, default: float) -> float:
    return float(os.getenv(env, default))


# --- Data locations -------------------------------------------------------
DATA_DIR = _path("BTS3D_DATA_DIR", PROJECT_ROOT / "data_3d")
TASK = "Task01_BrainTumour"
DATASET_DIR = DATA_DIR / TASK
DATASET_JSON = DATASET_DIR / "dataset.json"

# --- Output locations (git-ignored) ---------------------------------------
OUTPUT_DIR = _path("BTS3D_OUTPUT_DIR", PROJECT_ROOT / "outputs_3d")
MODELS_DIR = OUTPUT_DIR / "models"

# --- Problem definition ---------------------------------------------------
IN_CHANNELS = 4           # MRI modalities: FLAIR, T1w, T1Gd, T2w
OUT_CHANNELS = 3          # overlapping tumor regions, in this order:
REGIONS = ["TC", "WT", "ET"]  # tumor core, whole tumor, enhancing tumor

# --- Hyperparameters ------------------------------------------------------
PATCH_SIZE = (
    _int("BTS3D_PATCH", 96),
    _int("BTS3D_PATCH", 96),
    _int("BTS3D_PATCH", 96),
)
SPACING = (1.0, 1.0, 1.0)  # isotropic resample (mm)
BATCH_SIZE = _int("BTS3D_BATCH", 2)
EPOCHS = _int("BTS3D_EPOCHS", 100)
LEARNING_RATE = _float("BTS3D_LR", 1e-4)
SEED = _int("BTS3D_SEED", 42)
VAL_FRACTION = _float("BTS3D_VAL_FRACTION", 0.15)
TEST_FRACTION = _float("BTS3D_TEST_FRACTION", 0.15)
CACHE_RATE = _float("BTS3D_CACHE_RATE", 0.0)  # fraction of volumes cached in RAM


def ensure_output_dirs() -> None:
    for directory in (OUTPUT_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
