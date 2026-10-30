import pytest

from brain_tumor_segmentation.model import build_model, build_unet

INPUT_SHAPE = (128, 128, 3)


def test_scratch_unet_output_shape():
    model = build_model("scratch", INPUT_SHAPE)
    assert model.output_shape == (None, 128, 128, 1)


def test_pretrained_unet_output_shape_no_download():
    # weights=None avoids downloading ImageNet weights in CI.
    model = build_unet(INPUT_SHAPE, "resnet", weights=None)
    assert model.output_shape == (None, 128, 128, 1)


def test_unknown_encoder_raises():
    with pytest.raises(ValueError):
        build_unet(INPUT_SHAPE, "not_a_real_encoder", weights=None)
