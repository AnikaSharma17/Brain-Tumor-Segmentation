"""Load trained .h5 models and render prediction grids on the held-out test set."""

import keras

from brain_tumor_segmentation import config, data
from brain_tumor_segmentation.metrics import dice_coefficient
from brain_tumor_segmentation.visualization import visualize_segmentation


def load_trained_model(model_path):
    # compile=False: inference only, so the training loss/metrics need not be rebuilt.
    return keras.models.load_model(
        model_path, compile=False, custom_objects={"dice_coefficient": dice_coefficient})


def main():
    data.set_global_seed()
    config.ensure_output_dirs()
    # Same disjoint split as training -> genuinely held-out visualizations.
    _, _, test_ds = data.get_splits(config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

    for encoder_type in config.MODELS:
        model_path = config.MODELS_DIR / f"unet_{encoder_type}_model.h5"
        print(f"Loading model: {model_path}")
        if not model_path.exists():
            print(f"  Skipping {encoder_type}: {model_path} not found (run train.py first).")
            continue
        try:
            model = load_trained_model(str(model_path))
            save_path = config.PREDICTIONS_DIR / f"unet_{encoder_type}_predictions_loaded.png"
            visualize_segmentation(model, test_ds,
                                   f"U-Net with {encoder_type} Encoder (Loaded)",
                                   str(save_path), num_examples=3)
            print(f"  Saved visualization to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Error with {encoder_type}: {exc}")

    print(f"\nVisualizations saved under {config.PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
