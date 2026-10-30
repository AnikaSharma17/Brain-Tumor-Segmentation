"""Run a trained model on a single image and save the predicted mask + overlay.

Example:
    python scripts/predict.py --model outputs/models/unet_resnet_model.h5 \
        --image image_dataset/images/1.png --out prediction.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brain_tumor_segmentation import inference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict a tumor mask for one image.")
    parser.add_argument("--model", required=True, help="Path to a trained .h5 model.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--out", default="prediction.png", help="Output figure path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = inference.load_model(args.model)
    image = inference.load_image(args.image)
    prob, mask = inference.predict_mask(model, image, threshold=args.threshold)

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
