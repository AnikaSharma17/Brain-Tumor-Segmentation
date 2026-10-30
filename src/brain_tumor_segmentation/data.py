"""Data loading, splitting and augmentation.

Images and masks are paired explicitly by filename, split into disjoint
train/val/test subsets with a fixed seed, and masks are resized with
nearest-neighbour interpolation and re-binarized.

NOTE ON SPLITTING: the Figshare-style filenames (1.png ... 3064.png) do not expose
patient IDs, so this is a slice-level random split. If patient IDs become available,
switch to a patient-grouped split to remove residual leakage.
"""

from __future__ import annotations

import glob
import os
import random

import tensorflow as tf

from . import config


def set_global_seed(seed: int | None = None) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    tf.keras.utils.set_random_seed(config.SEED if seed is None else seed)


def list_pairs(
    image_dir: str | os.PathLike | None = None,
    mask_dir: str | os.PathLike | None = None,
) -> list[tuple[str, str]]:
    """Return [(image_path, mask_path), ...] paired explicitly by filename."""
    image_dir = str(image_dir or config.IMAGE_DIR)
    mask_dir = str(mask_dir or config.MASK_DIR)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    pairs, missing = [], 0
    for image_path in image_paths:
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))
        else:
            missing += 1
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found. Checked images in {image_dir!r} and "
            f"masks in {mask_dir!r}."
        )
    if missing:
        print(f"Warning: {missing} image(s) had no matching mask and were skipped.")
    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    seed: int | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split a pair list into disjoint (train, val, test) subsets."""
    seed = config.SEED if seed is None else seed
    val_frac = config.VAL_FRACTION if val_frac is None else val_frac
    test_frac = config.TEST_FRACTION if test_frac is None else test_frac
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def _load_pair(
    image_path: tf.Tensor, mask_path: tf.Tensor, img_height: int, img_width: int
) -> tuple[tf.Tensor, tf.Tensor]:
    # Source MRI slices are single-channel grayscale; decoding with channels=3
    # replicates the one channel into RGB so the ImageNet-pretrained encoders
    # (which expect 3 channels) can be used.
    image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    # Keep images in [0, 255]; the correct per-encoder preprocessing is applied
    # inside each model (see model.build_unet), not here.
    image = tf.image.resize(image, (img_height, img_width), method="bilinear")
    image = tf.cast(image, tf.float32)

    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    # Nearest-neighbour keeps masks binary through the resize, then re-binarize.
    mask = tf.image.resize(mask, (img_height, img_width), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Paired augmentation: geometric transforms are applied identically to the
    image and mask; noise is applied to the image only."""
    if tf.random.uniform(()) < 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) < 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    image = image + tf.random.normal(tf.shape(image), mean=0.0, stddev=5.0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mask


def make_dataset(
    pairs: list[tuple[str, str]],
    img_height: int | None = None,
    img_width: int | None = None,
    batch_size: int | None = None,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 123,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset from a list of path pairs."""
    img_height = config.IMG_HEIGHT if img_height is None else img_height
    img_width = config.IMG_WIDTH if img_width is None else img_width
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size

    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda i, m: _load_pair(i, m, img_height, img_width),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_splits(
    img_height: int | None = None,
    img_width: int | None = None,
    batch_size: int | None = None,
    augment_train: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return (train_ds, val_ds, test_ds) with disjoint samples.

    Augmentation (flips, rotations, noise) is applied to the training set only;
    validation and test are evaluated on clean data.
    """
    train, val, test = split_pairs(list_pairs())
    print(f"Dataset split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    train_ds = make_dataset(train, img_height, img_width, batch_size,
                            shuffle=True, augment=augment_train)
    val_ds = make_dataset(val, img_height, img_width, batch_size, shuffle=False)
    test_ds = make_dataset(test, img_height, img_width, batch_size, shuffle=False)
    return train_ds, val_ds, test_ds
