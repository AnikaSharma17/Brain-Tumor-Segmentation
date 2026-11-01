"""Retrain one model (or several) and refresh only their artifacts.

Unlike ``scripts/train.py`` (which always trains every model), this lets you pick
which model(s) to retrain and update in place: the saved model file, training curves,
CSV log, prediction grid, and that model's row in the results spreadsheet. The other
models' artifacts are left untouched.

Examples:
    python scripts/retrain.py --model efficientnet
    python scripts/retrain.py --model resnet vgg --lr-schedule cosine --epochs 30
    python scripts/retrain.py --model all

The EfficientNet BatchNorm fix lives in ``model.build_encoder`` and is applied
automatically, so retraining EfficientNet here reproduces the converged result.
"""

# Keras is imported directly rather than via `tensorflow.keras`: TF 2.15 installs
# Keras 2.15 as its own package, and `tensorflow.keras` is only a lazy alias to it
# that editors and linters can't resolve. Both names refer to the same objects.
import argparse

import keras
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam

from brain_tumor_segmentation import config, data
from brain_tumor_segmentation.losses import bce_dice_loss
from brain_tumor_segmentation.metrics import dice_coefficient
from brain_tumor_segmentation.model import build_model
from brain_tumor_segmentation.visualization import plot_history, visualize_segmentation


def _make_optimizer(schedule: str, epochs: int, steps_per_epoch: int):
    """Build the optimizer, optionally with a cosine-decay learning-rate schedule.

    'plateau' keeps a flat rate and is paired with a ReduceLROnPlateau callback;
    'cosine' decays the rate smoothly to zero over the whole run; 'none' is constant.
    """
    if schedule == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=max(1, steps_per_epoch * epochs))
        return Adam(learning_rate=lr)
    return Adam(learning_rate=config.LEARNING_RATE)


def _callbacks(name: str, schedule: str):
    cbs = [
        # Stop once val_loss stalls for 5 epochs and roll back to the best epoch.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(str(config.HISTORY_DIR / f"unet_{name}_log.csv")),
    ]
    if schedule == "plateau":
        cbs.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6))
    return cbs


def retrain_one(name, train_ds, val_ds, test_ds, input_shape, epochs, schedule):
    """Retrain a single model and return its held-out test metrics as a dict."""
    print(f"\nRetraining U-Net ({name}) "
          f"[epochs={epochs}, lr-schedule={schedule}]...")
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    model = build_model(name, input_shape=input_shape)
    model.compile(
        optimizer=_make_optimizer(schedule, epochs, steps_per_epoch),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient,
                 keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.BinaryIoU(target_class_ids=[0, 1],
                                         threshold=0.5, name="mean_iou")],
    )

    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                        callbacks=_callbacks(name, schedule), verbose=2)
    plot_history(history, name, str(config.HISTORY_DIR / f"unet_{name}_curves.png"))

    # evaluate() returns the loss followed by the metrics in compile() order.
    _, test_acc, test_dice, test_prec, test_rec, test_iou = model.evaluate(test_ds)

    model.save(str(config.model_path(name)))
    visualize_segmentation(
        model, test_ds, f"U-Net with {name} Encoder",
        str(config.PREDICTIONS_DIR / f"unet_{name}_predictions.png"), num_examples=3)
    print(f"Retrained {name}: test Dice={test_dice:.4f}, IoU={test_iou:.4f}, "
          f"precision={test_prec:.4f}, recall={test_rec:.4f}")
    return {"Model": name, "Accuracy": test_acc, "Dice Coefficient": test_dice,
            "Precision": test_prec, "Recall": test_rec, "Mean IoU": test_iou}


def update_results(rows):
    """Replace the given models' rows in the results spreadsheet, creating it if needed."""
    if config.RESULTS_XLSX.exists():
        df = pd.read_excel(config.RESULTS_XLSX)
    else:
        df = pd.DataFrame(columns=["Model", "Accuracy", "Dice Coefficient",
                                   "Precision", "Recall", "Mean IoU"])
    names = {r["Model"] for r in rows}
    df = df[~df["Model"].isin(names)]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # Keep the canonical model order for a stable, readable spreadsheet.
    order = {name: i for i, name in enumerate(config.MODELS)}
    df = df.sort_values("Model", key=lambda s: s.map(order)).reset_index(drop=True)
    df.to_excel(config.RESULTS_XLSX, index=False)
    print(f"\nUpdated results at {config.RESULTS_XLSX}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True,
                        choices=[*config.MODELS, "all"],
                        help="Model(s) to retrain, or 'all'.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Training epochs (default: {config.EPOCHS}).")
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "plateau"],
                        default="none", help="Learning-rate schedule (default: none).")
    args = parser.parse_args()

    names = config.MODELS if "all" in args.model else list(dict.fromkeys(args.model))

    data.set_global_seed()
    config.ensure_output_dirs()
    train_ds, val_ds, test_ds = data.get_splits(
        config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)

    rows = [retrain_one(name, train_ds, val_ds, test_ds, input_shape,
                        args.epochs, args.lr_schedule) for name in names]
    update_results(rows)


if __name__ == "__main__":
    main()
