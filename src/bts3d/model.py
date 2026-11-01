"""3D segmentation model builders (MONAI).

Two baselines: a plain 3D U-Net and SegResNet (a residual encoder-decoder that often
trains a little more stably on BraTS). Both take 4 input channels (MRI modalities) and
output 3 channels (the overlapping TC/WT/ET regions, scored with a sigmoid).
"""

from __future__ import annotations

from monai.networks.nets import SegResNet, UNet

from . import config


def build_model(name: str = "unet"):
    """Return a 3D segmentation network ('unet' or 'segresnet')."""
    if name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    raise ValueError(f"Unknown model: {name!r} (expected 'unet' or 'segresnet').")
