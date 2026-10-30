"""Train and evaluate all U-Net encoder variants, then save metrics and plots."""

import pandas as pd
import tensorflow as tf
# Import from standalone Keras: TF 2.16+ ships Keras 3, and `tensorflow.keras`
# is a lazy alias that editors and linters can't resolve.
from keras.optimizers import Adam

from brain_tumor_segmentation import config, data
from brain_tumor_segmentation.losses import bce_dice_loss
from brain_tumor_segmentation.metrics import dice_coefficient
from brain_tumor_segmentation.model import build_model
from brain_tumor_segmentation.visualization import plot_history, visualize_segmentation


def main():
    data.set_global_seed()
    config.ensure_output_dirs()

    train_ds, val_ds, test_ds = data.get_splits(
        config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    results = {"Model": [], "Accuracy": [], "Dice Coefficient": [],
               "Precision": [], "Recall": [], "Mean IoU": []}

    for encoder_type in config.MODELS:
        print(f"\nTraining U-Net ({encoder_type})...")
        model = build_model(encoder_type, input_shape=input_shape)
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss=bce_dice_loss,
            metrics=["accuracy", dice_coefficient,
                     tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                     tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1],
                                                threshold=0.5, name="mean_iou")],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                             restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(
                str(config.HISTORY_DIR / f"unet_{encoder_type}_log.csv")),
        ]
        history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds,
                            callbacks=callbacks, verbose=2)
        plot_history(history, encoder_type,
                     str(config.HISTORY_DIR / f"unet_{encoder_type}_curves.png"))

        test_loss, test_acc, test_dice, test_prec, test_rec, test_iou = model.evaluate(test_ds)
        results["Model"].append(encoder_type)
        results["Accuracy"].append(test_acc)
        results["Dice Coefficient"].append(test_dice)
        results["Precision"].append(test_prec)
        results["Recall"].append(test_rec)
        results["Mean IoU"].append(test_iou)

        model.save(str(config.MODELS_DIR / f"unet_{encoder_type}_model.h5"))
        visualize_segmentation(
            model, test_ds, f"U-Net with {encoder_type} Encoder",
            str(config.PREDICTIONS_DIR / f"unet_{encoder_type}_predictions.png"),
            num_examples=3)
        print(f"Saved model, curves and predictions for {encoder_type}.")

    pd.DataFrame(results).to_excel(config.RESULTS_XLSX, index=False)
    print(f"\nResults saved to {config.RESULTS_XLSX}")


if __name__ == "__main__":
    main()
