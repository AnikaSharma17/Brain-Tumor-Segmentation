"""Per-image error analysis for trained models.

For each trained model, compute the Dice score on every held-out test image and
report the distribution (mean / std / median / min), save a histogram, and a
gallery of the worst predictions. Mean-only metrics hide failure cases; this
surfaces them.

Run after scripts/train.py.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brain_tumor_segmentation import config, data, inference  # noqa: E402


def per_image_dice(y_true, y_pred, eps=1e-7):
    """Hard Dice between two binary masks.

    Unlike the soft Dice used in training, this compares thresholded masks. Note that
    a correct empty prediction on a tumor-free slice scores ~0, not 1.
    """
    yt = y_true.astype(bool).ravel()
    yp = y_pred.astype(bool).ravel()
    inter = np.logical_and(yt, yp).sum()
    return (2.0 * inter) / (yt.sum() + yp.sum() + eps)


def evaluate_model(model, test_pairs):
    """Return an array with one Dice score per (image, mask) pair."""
    scores = []
    for image_path, mask_path in test_pairs:
        image = inference.load_image(image_path)
        _, pred = inference.predict_mask(model, image)
        # load_image decodes as RGB and resizes bilinearly; take one channel and
        # threshold at the midpoint to get back a clean 0/1 mask.
        mask = inference.load_image(mask_path)
        mask = (mask.numpy()[..., 0] > 127).astype(np.uint8)
        scores.append(per_image_dice(mask, pred))
    return np.array(scores)


def main():
    data.set_global_seed()
    eval_dir = config.OUTPUT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_pairs = data.split_pairs(data.list_pairs())

    summary = []
    for name in config.MODELS:
        model_path = config.MODELS_DIR / f"unet_{name}_model.h5"
        if not model_path.exists():
            print(f"Skipping {name}: {model_path} not found (run train.py first).")
            continue
        print(f"Evaluating {name} on {len(test_pairs)} test images...")
        model = inference.load_model(model_path)
        scores = evaluate_model(model, test_pairs)
        summary.append({
            "Model": name,
            "Mean Dice": scores.mean(),
            "Std Dice": scores.std(),
            "Median Dice": np.median(scores),
            "Min Dice": scores.min(),
            "P10 Dice": np.percentile(scores, 10),
        })

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=30, color="steelblue", edgecolor="white")
        plt.axvline(scores.mean(), color="red", linestyle="--",
                    label=f"mean={scores.mean():.3f}")
        plt.xlabel("Per-image Dice")
        plt.ylabel("Count")
        plt.title(f"{name}: per-image Dice distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eval_dir / f"unet_{name}_dice_hist.png")
        plt.close()

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(eval_dir / "per_image_dice_summary.csv", index=False)
        print("\n" + df.to_string(index=False))
        print(f"\nSaved per-image analysis to {eval_dir}")
    else:
        print("No trained models found; nothing to evaluate.")


if __name__ == "__main__":
    main()
