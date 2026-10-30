"""U-Net model with selectable pretrained encoder backbones."""

from __future__ import annotations

from tensorflow.keras.applications import (
    VGG16,
    EfficientNetB0,
    MobileNetV2,
    ResNet50,
    efficientnet,
    mobilenet_v2,
    resnet50,
    vgg16,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    Lambda,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model

# Per-encoder ImageNet preprocessing. Feeding [0, 1] to these backbones is wrong:
# ResNet50/VGG16 expect caffe-style BGR mean subtraction, MobileNetV2 expects
# [-1, 1], and EfficientNetB0 expects raw [0, 255]. The matching preprocess_input is
# applied inside each model so it also travels with the saved weights.
PREPROCESS = {
    "resnet": resnet50.preprocess_input,
    "mobilenet": mobilenet_v2.preprocess_input,
    "vgg": vgg16.preprocess_input,
    "efficientnet": efficientnet.preprocess_input,
}

# Encoder -> skip-connection layer names (shallow to deep).
_SKIP_LAYERS = {
    "resnet": ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"],
    "mobilenet": ["block_1_expand_relu", "block_3_expand_relu",
                  "block_6_expand_relu", "block_13_expand_relu"],
    "vgg": ["block1_pool", "block2_pool", "block3_pool", "block4_pool"],
    "efficientnet": ["block2a_expand_activation", "block3a_expand_activation",
                     "block4a_expand_activation", "block6a_expand_activation"],
}

_BACKBONES = {
    "resnet": ResNet50,
    "mobilenet": MobileNetV2,
    "vgg": VGG16,
    "efficientnet": EfficientNetB0,
}


def build_encoder(
    input_shape: tuple[int, int, int],
    encoder_type: str,
    weights: str | None = "imagenet",
    freeze_bn: bool | None = None,
) -> Model:
    """Build a pretrained backbone that outputs the four skip-connection tensors.

    ``freeze_bn`` keeps the backbone's BatchNorm layers in inference mode (using the
    pretrained ImageNet moving statistics) instead of updating them during training.
    EfficientNet in Keras collapses at inference without this: its BatchNorm moving
    averages drift away from the batch statistics while fine-tuning, so training
    metrics look healthy but validation/inference predicts all-background and
    ``val_loss`` explodes. Defaults to freezing BN for EfficientNet only; the other
    backbones train fine with trainable BN and are left unchanged.
    """
    if encoder_type not in _BACKBONES:
        raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
    # weights=None is used in tests to build the graph without downloading weights.
    base_model = _BACKBONES[encoder_type](
        weights=weights, include_top=False, input_shape=input_shape)
    if freeze_bn is None:
        freeze_bn = encoder_type == "efficientnet"
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
    skip_connections = [base_model.get_layer(name).output
                        for name in _SKIP_LAYERS[encoder_type]]
    return Model(inputs=base_model.input, outputs=skip_connections)


def build_decoder(encoder_outputs: list):
    """Build the U-Net decoder (up-conv + skip concatenation) onto encoder outputs."""
    block1, block2, block3, block4 = encoder_outputs
    bottleneck = Conv2D(512, (3, 3), activation="relu", padding="same")(block4)

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(bottleneck)
    up3 = concatenate([up3, block3])
    up3 = Conv2D(256, (3, 3), activation="relu", padding="same")(up3)

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up3)
    up2 = concatenate([up2, block2])
    up2 = Conv2D(128, (3, 3), activation="relu", padding="same")(up2)

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up2)
    up1 = concatenate([up1, block1])
    up1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up1)

    outputs = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding="same")(up1)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(outputs)
    return outputs


def build_unet(
    input_shape: tuple[int, int, int], encoder_type: str, weights: str | None = "imagenet"
) -> Model:
    """Assemble a U-Net: pretrained encoder + decoder, with in-model preprocessing."""
    if encoder_type not in PREPROCESS:
        raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
    inputs = Input(input_shape)
    # Apply the encoder's matching ImageNet preprocessing to the raw [0, 255] input.
    preprocessed = Lambda(PREPROCESS[encoder_type],
                          name=f"{encoder_type}_preprocess")(inputs)
    encoder = build_encoder(input_shape, encoder_type, weights=weights)
    encoder_outputs = encoder(preprocessed)
    outputs = build_decoder(encoder_outputs)
    return Model(inputs=inputs, outputs=outputs)


def _conv_block(x, filters: int):
    """Two 3x3 same-padding ReLU convolutions, the standard U-Net double-conv block."""
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    return x


def build_unet_scratch(input_shape: tuple[int, int, int]) -> Model:
    """A plain U-Net trained from scratch (no pretrained encoder).

    Serves as the baseline the pretrained backbones should be compared against.
    """
    inputs = Input(input_shape)
    x = Lambda(lambda t: t / 255.0, name="scratch_rescale")(inputs)

    c1 = _conv_block(x, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = _conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = _conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = _conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    bottleneck = _conv_block(p4, 1024)

    u4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(bottleneck)
    u4 = _conv_block(concatenate([u4, c4]), 512)
    u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(u4)
    u3 = _conv_block(concatenate([u3, c3]), 256)
    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u3)
    u2 = _conv_block(concatenate([u2, c2]), 128)
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u2)
    u1 = _conv_block(concatenate([u1, c1]), 64)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u1)
    return Model(inputs=inputs, outputs=outputs)


def build_model(
    name: str, input_shape: tuple[int, int, int], weights: str | None = "imagenet"
) -> Model:
    """Dispatch: 'scratch' -> plain U-Net baseline, otherwise a pretrained encoder."""
    if name == "scratch":
        return build_unet_scratch(input_shape)
    return build_unet(input_shape, name, weights=weights)
