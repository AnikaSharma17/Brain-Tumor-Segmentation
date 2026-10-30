"""Retrain only the EfficientNetB0 U-Net and refresh its artifacts.

EfficientNet collapsed in the full run (val Dice ~0, exploding val_loss) due to the
Keras EfficientNet BatchNorm fine-tuning issue; the fix lives in
``model.build_encoder`` (BatchNorm frozen for EfficientNet). This script retrains
that one model with the fix in place and updates its .h5, training curves, CSV log,
prediction grid, and its row in the results spreadsheet, without retraining the
other four models. Re-running ``scripts/train.py`` reproduces the same fix.
"""

import keras
import pandas as pd
from keras.optimizers import Adam

from brain_tumor_segmentation import config, data
from brain_tumor_segmentation.losses import bce_dice_loss
from brain_tumor_segmentation.metrics import dice_coefficient
from brain_tumor_segmentation.model import build_model
from brain_tumor_segmentation.visualization import plot_history, visualize_segmentation

ENCODER = "efficientnet"


def main():
    data.set_global_seed()
    config.ensure_output_dirs()

    train_ds, val_ds, test_ds = data.get_splits(
        config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    print(f"\nRetraining U-Net ({ENCODER}) with BatchNorm frozen...")
    model = build_model(ENCODER, input_shape=input_shape)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient,
                 keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.BinaryIoU(target_class_ids=[0, 1],
                                         threshold=0.5, name="mean_iou")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(
            str(config.HISTORY_DIR / f"unet_{ENCODER}_log.csv")),
    ]
    history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds,
                        callbacks=callbacks, verbose=2)
    plot_history(history, ENCODER,
                 str(config.HISTORY_DIR / f"unet_{ENCODER}_curves.png"))

    # Unpack order must match the metrics list passed to compile() above.
    test_loss, test_acc, test_dice, test_prec, test_rec, test_iou = model.evaluate(test_ds)

    model.save(str(config.MODELS_DIR / f"unet_{ENCODER}_model.h5"))
    visualize_segmentation(
        model, test_ds, f"U-Net with {ENCODER} Encoder",
        str(config.PREDICTIONS_DIR / f"unet_{ENCODER}_predictions.png"),
        num_examples=3)

    # Update only the EfficientNet row in the existing results spreadsheet.
    df = pd.read_excel(config.RESULTS_XLSX)
    row = {"Model": ENCODER, "Accuracy": test_acc, "Dice Coefficient": test_dice,
           "Precision": test_prec, "Recall": test_rec, "Mean IoU": test_iou}
    df = df[df["Model"] != ENCODER]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(config.RESULTS_XLSX, index=False)

    print(f"\nRetrained {ENCODER}: test Dice={test_dice:.4f}, IoU={test_iou:.4f}, "
          f"precision={test_prec:.4f}, recall={test_rec:.4f}")
    print(f"Updated results at {config.RESULTS_XLSX}")


if __name__ == "__main__":
    main()
