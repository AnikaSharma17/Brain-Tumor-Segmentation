import numpy as np

from brain_tumor_segmentation import config, data


def _toy_pairs(n=100):
    return [(f"img_{i}.png", f"mask_{i}.png") for i in range(n)]


def test_split_is_disjoint_and_complete():
    pairs = _toy_pairs(100)
    train, val, test = data.split_pairs(pairs, seed=42, val_frac=0.15, test_frac=0.15)
    assert len(train) + len(val) + len(test) == len(pairs)
    s_tr, s_va, s_te = set(train), set(val), set(test)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)


def test_split_is_deterministic():
    pairs = _toy_pairs(50)
    a = data.split_pairs(pairs, seed=7)
    b = data.split_pairs(pairs, seed=7)
    assert a == b


def test_list_pairs_finds_real_dataset():
    if not config.IMAGE_DIR.exists():
        import pytest

        pytest.skip("dataset not present")
    pairs = data.list_pairs()
    assert len(pairs) > 0
    for image_path, mask_path in pairs[:5]:
        import os

        assert os.path.basename(image_path) == os.path.basename(mask_path)


def test_loaded_mask_is_binary():
    if not config.IMAGE_DIR.exists():
        import pytest

        pytest.skip("dataset not present")
    pairs = data.list_pairs()[:4]
    ds = data.make_dataset(pairs, 128, 128, batch_size=4)
    _, mask_batch = next(iter(ds))
    values = np.unique(mask_batch.numpy())
    assert set(values.tolist()).issubset({0.0, 1.0})
