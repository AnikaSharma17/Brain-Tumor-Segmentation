"""Plotting helpers: prediction grids and training curves."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def visualize_segmentation(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    model_name: str,
    save_path: str | os.PathLike,
    num_examples: int = 3,
) -> None:
    """Save an input / true-mask / predicted-mask grid for a few samples."""
    images, masks = [], []
    for image_batch, mask_batch in dataset:
        for j in range(image_batch.shape[0]):
            images.append(image_batch[j])
            masks.append(mask_batch[j])
            if len(images) >= num_examples:
                break
        if len(images) >= num_examples:
            break
    num_examples = len(images)
    predicted_masks = model.predict(tf.stack(images))

    plt.figure(figsize=(15, 5 * num_examples))
    plt.suptitle(f"Predictions from {model_name}", fontsize=16)
    for r in range(num_examples):
        plt.subplot(num_examples, 3, r * 3 + 1)
        plt.imshow(images[r].numpy().astype(np.uint8))  # images are in [0, 255]
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_examples, 3, r * 3 + 2)
        plt.imshow(masks[r].numpy().squeeze(), cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(num_examples, 3, r * 3 + 3)
        predicted_mask = (predicted_masks[r].squeeze() > 0.5).astype(np.uint8)
        plt.imshow(predicted_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_history(history, encoder_type: str, save_path: str | os.PathLike) -> None:
    """Save loss and Dice training curves for one model."""
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(epochs_range, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{encoder_type}: Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist["dice_coefficient"], label="train dice")
    if "val_dice_coefficient" in hist:
        plt.plot(epochs_range, hist["val_dice_coefficient"], label="val dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"{encoder_type}: Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
