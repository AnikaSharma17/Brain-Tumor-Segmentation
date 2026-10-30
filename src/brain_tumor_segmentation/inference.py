"""Inference helpers for running a trained model on a single image."""

from __future__ import annotations

import os

import keras
import numpy as np
import tensorflow as tf

from . import config
from .metrics import dice_coefficient


def load_model(model_path: str | os.PathLike) -> keras.Model:
    """Load a trained .h5 model for inference (no recompilation)."""
    return keras.models.load_model(
        str(model_path), compile=False,
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
