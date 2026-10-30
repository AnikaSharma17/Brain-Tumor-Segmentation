"""Central configuration.

Every value has a sensible default and can be overridden with an environment
variable (optionally via a local ``.env`` file; see ``.env.example``). This keeps
machine-specific paths and hyperparameters out of the code.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional; defaults work without it.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Repository root (…/src/brain_tumor_segmentation/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str | Path) -> Path:
    value = os.getenv(env)
    return Path(value) if value else Path(default)


def _int(env: str, default: int) -> int:
    return int(os.getenv(env, default))


def _float(env: str, default: float) -> float:
    return float(os.getenv(env, default))


# --- Data locations -------------------------------------------------------
DATA_DIR = _path("BTS_DATA_DIR", PROJECT_ROOT / "image_dataset")
IMAGE_DIR = _path("BTS_IMAGE_DIR", DATA_DIR / "images")
MASK_DIR = _path("BTS_MASK_DIR", DATA_DIR / "masks")

# --- Output locations (generated at runtime, git-ignored) -----------------
OUTPUT_DIR = _path("BTS_OUTPUT_DIR", PROJECT_ROOT / "outputs")
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
HISTORY_DIR = OUTPUT_DIR / "training_history"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
RESULTS_XLSX = OUTPUT_DIR / "model_comparison_results.xlsx"

# --- Hyperparameters ------------------------------------------------------
IMG_HEIGHT = _int("BTS_IMG_HEIGHT", 128)
IMG_WIDTH = _int("BTS_IMG_WIDTH", 128)
BATCH_SIZE = _int("BTS_BATCH_SIZE", 8)
EPOCHS = _int("BTS_EPOCHS", 20)
LEARNING_RATE = _float("BTS_LEARNING_RATE", 1e-4)
SEED = _int("BTS_SEED", 42)
VAL_FRACTION = _float("BTS_VAL_FRACTION", 0.15)
TEST_FRACTION = _float("BTS_TEST_FRACTION", 0.15)

# Encoder backbones.
ENCODERS = ["resnet", "mobilenet", "vgg", "efficientnet"]

# Models to train/compare: a from-scratch U-Net baseline plus each pretrained encoder.
MODELS = ["scratch", *ENCODERS]


def ensure_output_dirs() -> None:
    """Create all output directories if they do not exist."""
    for directory in (OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, HISTORY_DIR, GRAPHS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
