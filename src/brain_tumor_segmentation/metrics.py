"""Segmentation metrics."""

from __future__ import annotations

import keras
import tensorflow as tf


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice coefficient over the whole batch."""
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    denom = keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f)
    return (2.0 * intersection) / (denom + keras.backend.epsilon())
