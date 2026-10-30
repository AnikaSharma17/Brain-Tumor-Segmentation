import tensorflow as tf

from brain_tumor_segmentation.metrics import dice_coefficient


def test_dice_perfect_overlap():
    mask = tf.constant([[1.0, 1.0], [0.0, 0.0]])
    assert float(dice_coefficient(mask, mask)) > 0.999


def test_dice_no_overlap():
    y_true = tf.constant([[1.0, 1.0], [0.0, 0.0]])
    y_pred = tf.constant([[0.0, 0.0], [1.0, 1.0]])
    assert float(dice_coefficient(y_true, y_pred)) < 1e-3


def test_dice_empty_masks_is_zero():
    # Both empty: 2*0 / (0 + 0 + eps) -> ~0 (documented edge behaviour of soft Dice).
    zeros = tf.zeros((2, 2))
    assert float(dice_coefficient(zeros, zeros)) < 1e-3
