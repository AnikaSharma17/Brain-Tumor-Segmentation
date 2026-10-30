import tensorflow as tf

from brain_tumor_segmentation.losses import bce_dice_loss


def test_bce_dice_loss_is_finite_and_nonnegative():
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]])
    loss = float(bce_dice_loss(y_true, y_pred))
    assert loss >= 0.0
    assert loss == loss  # not NaN


def test_bce_dice_loss_lower_for_better_prediction():
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    good = tf.constant([[0.95, 0.05], [0.05, 0.95]])
    bad = tf.constant([[0.5, 0.5], [0.5, 0.5]])
    assert float(bce_dice_loss(y_true, good)) < float(bce_dice_loss(y_true, bad))
