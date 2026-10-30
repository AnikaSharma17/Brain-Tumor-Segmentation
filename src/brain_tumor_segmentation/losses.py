"""Loss functions."""

from __future__ import annotations

import keras
import tensorflow as tf

from .metrics import dice_coefficient


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Binary cross-entropy + (1 - Dice).

    Plain BCE is dominated by the large background class (tumors are a small
    fraction of each slice); the Dice term makes the loss care about foreground
    overlap.
    """
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coefficient(y_true, y_pred))
