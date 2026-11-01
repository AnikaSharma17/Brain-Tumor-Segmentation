"""3D data pipeline: MONAI transforms, patient-level split, and loaders.

Each MSD Task01 case is one patient (a 4-modality volume), so splitting the case list
gives a genuinely patient-level, leakage-free train/val/test split — unlike the 2D
project's slice-level split. Labels are converted to the three overlapping BraTS regions
(TC, WT, ET) with MONAI's standard transform.
"""

from __future__ import annotations

import random

from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

from . import config


def load_split(seed=None, val_frac=None, test_frac=None):
    """Patient-level train/val/test split of the MSD Task01 training cases."""
    if not config.DATASET_JSON.exists():
        raise FileNotFoundError(
            f"{config.DATASET_JSON} not found. Download the dataset first:\n"
            f"  python scripts/sanity_3d.py --download")
    seed = config.SEED if seed is None else seed
    val_frac = config.VAL_FRACTION if val_frac is None else val_frac
    test_frac = config.TEST_FRACTION if test_frac is None else test_frac
    items = load_decathlon_datalist(
        str(config.DATASET_JSON), is_segmentation=True,
        data_list_key="training", base_dir=str(config.DATASET_DIR))
    random.Random(seed).shuffle(items)
    n = len(items)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def train_transforms(patch=None):
    patch = config.PATCH_SIZE if patch is None else patch
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),  # (H,W,D,4) -> (4,H,W,D)
        # MSD Task01 / BraTS 2023 label it ET as 3 (not the default 4); without
        # et_label=3 the ET channel is empty and TC/WT silently drop enhancing voxels.
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=3),  # (3,H,W,D)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config.SPACING,
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ])


def eval_transforms():
    """Full-volume transforms for validation/test (no cropping; use sliding window)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=3),  # (3,H,W,D)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config.SPACING,
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_loaders(patch=None, batch_size=None, cache_rate=None, limit=None, num_workers=2):
    """Return (train_loader, val_loader, test_list). ``limit`` truncates each split
    (for quick smoke tests)."""
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    cache_rate = config.CACHE_RATE if cache_rate is None else cache_rate
    train, val, test = load_split()
    if limit:
        train, val, test = train[:limit], val[:max(1, limit // 2)], test[:max(1, limit // 2)]
    print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    train_ds = CacheDataset(train, transform=train_transforms(patch),
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(val, transform=eval_transforms(),
                          cache_rate=cache_rate, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test
