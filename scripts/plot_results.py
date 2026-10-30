"""Generate bar-chart comparisons from the training results spreadsheet."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from brain_tumor_segmentation import config  # noqa: E402

# Metric column -> (output filename, bar color)
METRICS = {
    "Accuracy": ("accuracy_comparison.png", "skyblue"),
    "Dice Coefficient": ("dice_coefficient_comparison.png", "lightcoral"),
    "Precision": ("precision_comparison.png", "lightgreen"),
    "Recall": ("recall_comparison.png", "gold"),
    "Mean IoU": ("mean_iou_comparison.png", "lightsalmon"),
}


def _ylim(values):
    """Zoom the y-axis onto the data (models often differ by <0.05), clamped to [0, 1]."""
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.25, 0.01)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def main():
    if not config.RESULTS_XLSX.exists():
        raise FileNotFoundError(
            f"{config.RESULTS_XLSX} not found. Run scripts/train.py first.")
    df = pd.read_excel(config.RESULTS_XLSX)
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    models = df["Model"].tolist()

    for metric, (filename, color) in METRICS.items():
        if metric not in df.columns:
            print(f"Skipping '{metric}' (not in results).")
            continue
        values = df[metric].tolist()
        plt.figure(figsize=(8, 6))
        plt.bar(models, values, color=color)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison of U-Net Models")
        plt.ylim(*_ylim(values))
        plt.grid(axis="y", linestyle="--")
        plt.tight_layout()
        plt.savefig(config.GRAPHS_DIR / filename)
        plt.close()

    print(f"Bar graphs saved to {config.GRAPHS_DIR}")


if __name__ == "__main__":
    main()
