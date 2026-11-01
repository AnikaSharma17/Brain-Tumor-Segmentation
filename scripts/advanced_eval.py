"""Advanced held-out evaluation: test-time augmentation (TTA) and model ensembling.

Two cheap, no-retraining ways to squeeze more accuracy out of already-trained models:

* TTA: average each model's predictions over the eight D4 symmetries of the input.
* Ensemble: average the probability maps of several models before thresholding.

For every selected model this reports per-image Dice with and without TTA, then the
ensemble (mean of the models' probability maps) with and without TTA, so the gains are
directly comparable to the single-model numbers from ``scripts/evaluate.py``.

Examples:
    python scripts/advanced_eval.py
    python scripts/advanced_eval.py --models resnet efficientnet vgg scratch mobilenet
"""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from brain_tumor_segmentation import config, data, inference

# The eight dihedral symmetries, shared with inference.predict_mask_tta.
_D4 = [(k, flip) for k in range(4) for flip in (False, True)]


def _tta_probs(model, x):
    """Average probability maps over the eight D4 transforms for a batch x (B,H,W,3)."""
    acc = None
    for k, flip in _D4:
        xt = tf.image.rot90(x, k)
        if flip:
            xt = tf.reverse(xt, axis=[2])
        p = model(xt, training=False)
        if flip:
            p = tf.reverse(p, axis=[2])
        p = tf.image.rot90(p, (4 - k) % 4)
        acc = p if acc is None else acc + p
    return acc / len(_D4)


def _per_image_dice(masks, probs, threshold=0.5, eps=1e-7):
    """Vectorized hard Dice per image between ground-truth masks and probability maps."""
    pred = (probs > threshold).reshape(probs.shape[0], -1)
    true = (masks > 0.5).reshape(masks.shape[0], -1)
    inter = np.logical_and(pred, true).sum(axis=1)
    return (2.0 * inter) / (pred.sum(axis=1) + true.sum(axis=1) + eps)


def _stats(name, scores):
    return {"Config": name, "Mean Dice": scores.mean(), "Std Dice": scores.std(),
            "Median Dice": np.median(scores), "P10 Dice": np.percentile(scores, 10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["resnet", "efficientnet", "vgg"],
                        choices=config.MODELS, help="Models to evaluate and ensemble.")
    args = parser.parse_args()

    data.set_global_seed()
    eval_dir = config.OUTPUT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_ds = data.get_splits(config.IMG_HEIGHT, config.IMG_WIDTH, config.BATCH_SIZE)

    # Collect ground-truth masks once, in the dataset's fixed (unshuffled) order.
    masks = np.concatenate([m.numpy()[..., 0] for _, m in test_ds], axis=0)

    summary = []
    ens_base = np.zeros_like(masks, dtype=np.float32)
    ens_tta = np.zeros_like(masks, dtype=np.float32)
    n_models = 0

    for name in args.models:
        model_path = config.model_path(name)
        if not model_path.exists():
            print(f"Skipping {name}: {model_path} not found (train it first).")
            continue
        print(f"Evaluating {name} (baseline + TTA) on {masks.shape[0]} test images...")
        model = inference.load_model(model_path)
        base_batches, tta_batches = [], []
        for x, _ in test_ds:
            base_batches.append(model(x, training=False).numpy()[..., 0])
            tta_batches.append(_tta_probs(model, x).numpy()[..., 0])
        base = np.concatenate(base_batches, axis=0)
        tta = np.concatenate(tta_batches, axis=0)

        summary.append(_stats(f"{name}", _per_image_dice(masks, base)))
        summary.append(_stats(f"{name} + TTA", _per_image_dice(masks, tta)))
        ens_base += base
        ens_tta += tta
        n_models += 1

    if n_models >= 2:
        label = "+".join(args.models)
        summary.append(_stats(f"ensemble[{label}]",
                              _per_image_dice(masks, ens_base / n_models)))
        summary.append(_stats(f"ensemble[{label}] + TTA",
                              _per_image_dice(masks, ens_tta / n_models)))
    elif n_models == 1:
        print("Only one model available; skipping ensemble.")

    if summary:
        df = pd.DataFrame(summary)
        out_csv = eval_dir / "advanced_eval_summary.csv"
        df.to_csv(out_csv, index=False)
        print("\n" + df.to_string(index=False))
        print(f"\nSaved to {out_csv}")
    else:
        print("No trained models found; nothing to evaluate.")


if __name__ == "__main__":
    main()
