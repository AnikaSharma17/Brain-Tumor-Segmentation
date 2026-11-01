"""Train a baseline 3D U-Net / SegResNet on MSD Task01 (PyTorch + MONAI).

Patch-based training with mixed precision, sliding-window validation, and per-region
Dice (TC / WT / ET). Designed to fit an 8 GB GPU: 96^3 patches, small batch, AMP.

Examples:
    python scripts/train_3d.py                       # full run (see bts3d.config)
    python scripts/train_3d.py --model segresnet --epochs 150
    python scripts/train_3d.py --smoke               # 1 epoch on a few volumes (sanity)

Run with the 3D environment's interpreter (.venv-3d).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The bts3d package lives under src/ and isn't pip-installed in .venv-3d, so add it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism

from bts3d import config, data
from bts3d.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unet", choices=["unet", "segresnet"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patch", type=int, default=config.PATCH_SIZE[0])
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--loss", default="dicefocal", choices=["dicefocal", "dicece"],
                        help="dicefocal up-weights the rare ET voxels (fixes ET collapse); "
                             "dicece is the plain Dice + cross-entropy baseline.")
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "none"],
                        help="Cosine-anneal the LR to 0 over training (helps late convergence).")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (1 epoch, few volumes) to verify the pipeline.")
    args = parser.parse_args()

    set_determinism(seed=config.SEED)
    config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    patch = (args.patch,) * 3

    epochs = 1 if args.smoke else args.epochs
    val_interval = 1 if args.smoke else args.val_interval
    limit = 6 if args.smoke else None
    train_loader, val_loader, _ = data.get_loaders(
        patch=patch, batch_size=args.batch, limit=limit)

    model = build_model(args.model).to(device)
    if args.loss == "dicefocal":
        # Focal term concentrates on hard/rare voxels -> recovers the tiny ET region.
        loss_fn = DiceFocalLoss(sigmoid=True, to_onehot_y=False,
                                lambda_dice=1.0, lambda_focal=1.0)
    else:
        loss_fn = DiceCELoss(sigmoid=True, to_onehot_y=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=1e-5)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if args.lr_schedule == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Multi-label post-processing + per-region Dice (all 3 channels are foreground).
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    print(f"device={device}, amp={amp}, model={args.model}, patch={patch}, "
          f"batch={args.batch}, epochs={epochs}")
    best_mean = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, 1):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{epochs}  loss={epoch_loss / step:.4f}  "
              f"lr={lr:.2e}  ({time.time() - t0:.0f}s)")

        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    with torch.amp.autocast("cuda", enabled=amp):
                        logits = sliding_window_inference(x, patch, 1, model, overlap=0.5)
                    preds = [post(p) for p in decollate_batch(logits)]
                    dice_metric(y_pred=preds, y=decollate_batch(y))
            scores = dice_metric.aggregate().tolist()  # per region: TC, WT, ET
            dice_metric.reset()
            mean = sum(scores) / len(scores)
            per_region = "  ".join(f"{r}={s:.4f}"
                                   for r, s in zip(config.REGIONS, scores))
            print(f"  val Dice  mean={mean:.4f}  {per_region}")
            if mean > best_mean:
                best_mean = mean
                out = config.MODELS_DIR / f"bts3d_{args.model}_best.pt"
                torch.save(model.state_dict(), out)
                print(f"  saved best -> {out}")

    print(f"\nDone. Best mean Dice: {best_mean:.4f}")


if __name__ == "__main__":
    main()
