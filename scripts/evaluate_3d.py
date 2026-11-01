"""Evaluate a trained 3D model on the held-out MSD Task01 TEST split.

Reports per-region Dice and Hausdorff95 (TC / WT / ET) on the patient-level test
split that training never saw, and saves a worst-case gallery (the lowest mean-Dice
cases, each shown as a middle tumor slice with ground-truth vs predicted regions).

Dice is nan-averaged across cases (a case with no ground-truth voxels for a region is
skipped for that region, the standard BraTS convention). HD95 is reported in voxels at
the resampled 1 mm isotropic spacing, averaged over cases where it is defined.

Example:
    python scripts/evaluate_3d.py --model unet \
        --checkpoint outputs_3d/models/bts3d_unet_best.pt

Run with the 3D environment's interpreter (.venv-3d).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The bts3d package lives under src/ and isn't pip-installed in .venv-3d, so add it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism

from bts3d import config, data
from bts3d.model import build_model


def build_test_loader(limit=None, num_workers=2):
    """Loader over the held-out test split (full volumes, no cropping)."""
    _, _, test = data.load_split()
    if limit:
        test = test[:limit]
    print(f"Test cases: {len(test)}")
    ds = CacheDataset(test, transform=data.eval_transforms(), cache_rate=0.0,
                      num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers), test


def region_dice(pred, gt):
    """Per-region Dice for one case. pred/gt: (C, H, W, D) binary tensors.

    Returns a list with one value per region; ``nan`` where the ground truth for
    that region is empty (undefined Dice -> excluded from the mean, BraTS-style)."""
    dices = []
    for c in range(pred.shape[0]):
        p = pred[c].reshape(-1).float()
        g = gt[c].reshape(-1).float()
        denom = p.sum() + g.sum()
        if g.sum() == 0:
            dices.append(float("nan"))
        else:
            dices.append((2.0 * (p * g).sum() / denom).item())
    return dices


def _best_slice(gt_wt):
    """Index of the axial slice with the most whole-tumor voxels (for display)."""
    areas = gt_wt.sum(axis=(0, 1))  # sum over H, W -> per-D
    return int(areas.argmax()) if areas.max() > 0 else gt_wt.shape[2] // 2


def _regions_rgb(mask3, z):
    """Overlay the three binary regions of one slice as RGB (TC=red, WT=green, ET=blue)."""
    sl = mask3[:, :, :, z]
    rgb = np.zeros(sl.shape[1:] + (3,), dtype=float)
    rgb[..., 0] = sl[0]  # TC
    rgb[..., 1] = sl[1]  # WT
    rgb[..., 2] = sl[2]  # ET
    return np.clip(rgb, 0, 1)


def save_gallery(cases, out_path):
    """cases: list of dicts with name, dice, flair, gt_rgb, pred_rgb (2D arrays)."""
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, case in enumerate(cases):
        axes[row][0].imshow(case["flair"].T, cmap="gray", origin="lower")
        axes[row][0].set_ylabel(f"{case['name']}\nmean Dice {case['dice']:.3f}",
                                fontsize=9)
        axes[row][1].imshow(np.transpose(case["gt_rgb"], (1, 0, 2)), origin="lower")
        axes[row][2].imshow(np.transpose(case["pred_rgb"], (1, 0, 2)), origin="lower")
        for col, title in enumerate(("FLAIR", "ground truth", "prediction")):
            if row == 0:
                axes[row][col].set_title(title)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
    fig.suptitle("Worst cases (TC=red, WT=green, ET=blue)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unet", choices=["unet", "segresnet"])
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt state_dict.")
    parser.add_argument("--patch", type=int, default=config.PATCH_SIZE[0])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid decision threshold for all regions.")
    parser.add_argument("--gallery-n", type=int, default=6,
                        help="Number of worst cases to render (0 to skip).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test cases (for a quick check).")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_determinism(seed=config.SEED)
    config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    patch = (args.patch,) * 3

    model = build_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint}")

    loader, _ = build_test_loader(limit=args.limit)
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=args.threshold)])
    hd95 = HausdorffDistanceMetric(include_background=True, percentile=95,
                                   reduction="mean_batch")

    dice_rows = []   # per-case list of [TC, WT, ET]
    gallery = []     # per-case display payload
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            name = Path(batch["image"].meta["filename_or_obj"][0]).name
            with torch.amp.autocast("cuda", enabled=amp):
                logits = sliding_window_inference(x, patch, 1, model, overlap=0.5)
            pred = post(decollate_batch(logits)[0])          # (3, H, W, D)
            gt = decollate_batch(y)[0]                        # (3, H, W, D)
            hd95(y_pred=[pred], y=[gt])

            dices = region_dice(pred, gt)
            dice_rows.append(dices)
            mean_d = float(np.nanmean(dices))

            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()
            z = _best_slice(gt_np[1])  # WT channel
            gallery.append({
                "name": name,
                "dice": mean_d,
                "flair": x[0, 0, :, :, z].cpu().numpy(),  # modality 0 = FLAIR
                "gt_rgb": _regions_rgb(gt_np, z),
                "pred_rgb": _regions_rgb(pred_np, z),
            })
            print(f"  {name:28s} mean={mean_d:.4f}  "
                  + "  ".join(f"{r}={d:.4f}" for r, d in zip(config.REGIONS, dices)))

    dice_arr = np.array(dice_rows, dtype=float)             # (N, 3)
    dice_per_region = np.nanmean(dice_arr, axis=0)
    hd_per_region = hd95.aggregate().cpu().numpy()
    hd_per_region = np.where(np.isfinite(hd_per_region), hd_per_region, np.nan)

    print("\n=== Held-out test set ===")
    print(f"{'region':>6}  {'Dice':>8}  {'HD95(vox)':>10}")
    for r, d, h in zip(config.REGIONS, dice_per_region, hd_per_region):
        print(f"{r:>6}  {d:>8.4f}  {h:>10.2f}")
    print(f"{'mean':>6}  {np.nanmean(dice_per_region):>8.4f}  "
          f"{np.nanmean(hd_per_region):>10.2f}")

    if args.gallery_n > 0:
        worst = sorted(gallery, key=lambda c: c["dice"])[:args.gallery_n]
        out = config.OUTPUT_DIR / f"test_worst_{args.model}.png"
        save_gallery(worst, out)
        print(f"\nWorst-case gallery -> {out}")


if __name__ == "__main__":
    main()
