"""Phase 0 sanity check for the 3D track.

Verifies that PyTorch sees the GPU and can actually run a kernel on it (important on
Blackwell / RTX 50-series, where torch.cuda.is_available() can be True but kernels fail
without cu128 wheels). With --download, it also pulls the MSD Task01_BrainTumour dataset
via MONAI, loads one volume, prints its shapes, and saves a slice montage.

Usage:
    python scripts/sanity_3d.py                 # GPU / PyTorch check only
    python scripts/sanity_3d.py --download      # also download + load one MSD volume (~7 GB)
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs_3d"


def check_gpu() -> bool:
    """Print PyTorch/CUDA status and run a tiny matmul on the GPU to prove kernels work."""
    import torch

    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device visible to PyTorch. Install the cu128 build:")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return False
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()  # forces a real kernel launch on the GPU
        print(f"GPU matmul OK (checksum={y:.1f})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"GPU kernel launch FAILED: {exc}")
        print("This is the classic Blackwell/sm_120 wheel mismatch; reinstall with cu128.")
        return False


def check_data(data_dir: Path) -> None:
    """Download MSD Task01 (if needed), load one volume, print shapes, save a montage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from monai.apps import DecathlonDataset
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

    data_dir.mkdir(parents=True, exist_ok=True)
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])
    print(f"\nDownloading / loading MSD Task01_BrainTumour into {data_dir} ...")
    ds = DecathlonDataset(root_dir=str(data_dir), task="Task01_BrainTumour",
                          section="training", transform=transform, download=True,
                          cache_num=0, num_workers=2)
    item = ds[0]
    image, label = item["image"], item["label"]
    print(f"image shape (C, H, W, D): {tuple(image.shape)}")  # 4 modalities expected
    print(f"label shape: {tuple(label.shape)}")
    print(f"label values: {np.unique(np.asarray(label))}")
    print(f"dataset size (training section): {len(ds)}")

    # Middle-depth slice: 4 modalities + the label.
    img = np.asarray(image)
    lab = np.asarray(label)[0]
    d = img.shape[-1] // 2
    names = ["FLAIR", "T1w", "T1Gd", "T2w"]  # MSD Task01 channel order
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for c in range(min(4, img.shape[0])):
        axes[c].imshow(img[c, :, :, d], cmap="gray")
        axes[c].set_title(names[c] if c < len(names) else f"mod {c}")
        axes[c].axis("off")
    axes[4].imshow(lab[:, :, d], cmap="viridis")
    axes[4].set_title("label")
    axes[4].axis("off")
    plt.tight_layout()
    out = OUTPUT_DIR / "sanity_montage.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved slice montage to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true",
                        help="Download MSD Task01 (~7 GB) and load one volume.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Dataset directory (default: {DEFAULT_DATA_DIR}).")
    args = parser.parse_args()

    gpu_ok = check_gpu()
    if args.download:
        check_data(args.data_dir)
    else:
        print("\n(Skipping data download; pass --download to fetch MSD Task01.)")
    print(f"\nGPU check: {'PASS' if gpu_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
