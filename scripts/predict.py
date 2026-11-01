"""Run a trained model on a single image and save the predicted mask + overlay.

Example:
    python scripts/predict.py --model outputs/models/unet_resnet_model.keras \
        --image image_dataset/images/1.png --out prediction.png

Add --tta to average predictions over the eight D4 symmetries (slower, usually a
little more accurate).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brain_tumor_segmentation import inference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict a tumor mask for one image.")
    parser.add_argument("--model", required=True,
                        help="Path to a trained model (.keras or .h5).")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--out", default="prediction.png", help="Output figure path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true",
                        help="Average predictions over the eight D4 symmetries.")
    args = parser.parse_args()

    model = inference.load_model(args.model)
    image = inference.load_image(args.image)
    predict = inference.predict_mask_tta if args.tta else inference.predict_mask
    prob, mask = predict(model, image, threshold=args.threshold)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(prob, cmap="magma")
    plt.title("Probability")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(image.numpy().astype("uint8"))
    plt.imshow(mask, cmap="Reds", alpha=0.4)
    plt.title("Predicted mask")
    plt.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    plt.close()
    print(f"Saved prediction to {args.out} "
          f"(tumor pixels: {int(mask.sum())}/{mask.size})")


if __name__ == "__main__":
    main()
