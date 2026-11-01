"""Inference helpers for running a trained model on a single image."""

from __future__ import annotations

import os

import keras
import numpy as np
import tensorflow as tf

from . import config
from .metrics import dice_coefficient


def load_model(model_path: str | os.PathLike) -> keras.Model:
    """Load a trained model (.keras or legacy .h5) for inference (no recompilation).

    ``safe_mode=False`` is required to deserialize the Lambda preprocessing layers
    baked into each model; these are our own layers, not untrusted content.
    """
    return keras.models.load_model(
        str(model_path), compile=False, safe_mode=False,
        custom_objects={"dice_coefficient": dice_coefficient})


def load_image(
    path: str | os.PathLike, img_height: int | None = None, img_width: int | None = None
) -> tf.Tensor:
    """Load and resize an image to model input size, kept in [0, 255] float32."""
    img_height = config.IMG_HEIGHT if img_height is None else img_height
    img_width = config.IMG_WIDTH if img_width is None else img_width
    image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
    image = tf.image.resize(image, (img_height, img_width), method="bilinear")
    return tf.cast(image, tf.float32)


def predict_mask(
    model: keras.Model, image: tf.Tensor, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability_map, binary_mask) for a single HxWx3 image."""
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)  # add the batch axis the model expects
    # Output is (1, H, W, 1); drop the batch and channel axes to get an HxW map.
    prob = model.predict(image, verbose=0)[0, ..., 0]
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


# The eight dihedral (D4) symmetries as (rotations, horizontal-flip) pairs: every
# 90-degree rotation, with and without a left-right flip. Each is a geometric
# transform whose prediction we map back before averaging.
_D4 = [(k, flip) for k in range(4) for flip in (False, True)]


def predict_mask_tta(
    model: keras.Model, image: tf.Tensor, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Test-time augmentation: average predictions over the eight D4 symmetries.

    The input is transformed, predicted, and the prediction is mapped back to the
    original orientation before averaging. Costs 8x inference but typically lifts Dice
    a point or two for free (no retraining). Returns (probability_map, binary_mask).
    """
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)  # (1, H, W, 3)
    acc = None
    for k, flip in _D4:
        x = tf.image.rot90(image, k)
        if flip:
            x = tf.reverse(x, axis=[2])  # flip width axis
        p = model(x, training=False)  # (1, H, W, 1)
        if flip:
            p = tf.reverse(p, axis=[2])
        p = tf.image.rot90(p, (4 - k) % 4)  # undo the rotation
        acc = p if acc is None else acc + p
    prob = (acc / len(_D4)).numpy()[0, ..., 0]
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask
