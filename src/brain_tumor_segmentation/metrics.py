"""Segmentation metrics."""

from __future__ import annotations

import tensorflow as tf


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice coefficient over the whole batch."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    denom = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2.0 * intersection) / (denom + tf.keras.backend.epsilon())
