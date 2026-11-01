"""Load trained models and render prediction grids on the held-out test set."""

from brain_tumor_segmentation import config, data, inference
from brain_tumor_segmentation.visualization import visualize_segmentation


def main():
    data.set_global_seed()
    config.ensure_output_dirs()
    # Same disjoint split as training -> genuinely held-out visualizations.
    _, _, test_ds = data.get_splits(config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

    for encoder_type in config.MODELS:
        model_path = config.model_path(encoder_type)
        print(f"Loading model: {model_path}")
        if not model_path.exists():
            print(f"  Skipping {encoder_type}: {model_path} not found (run train.py first).")
            continue
        try:
            model = inference.load_model(str(model_path))
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
