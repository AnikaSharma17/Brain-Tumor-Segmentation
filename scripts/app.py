"""Optional Gradio demo: upload an MRI slice and see the predicted tumor mask.

Requires the demo extra:  pip install -e ".[demo]"
Run:                      python scripts/app.py --model outputs/models/unet_resnet_model.h5
"""

import argparse

import numpy as np
import tensorflow as tf

from brain_tumor_segmentation import config, inference

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'Gradio is not installed. Install the demo extra:  pip install -e ".[demo]"'
    ) from exc


def build_interface(model):
    def segment(image):
        img = tf.convert_to_tensor(image, dtype=tf.float32)
        if img.ndim == 2:  # a 2-D grayscale upload has no channel axis
            img = tf.expand_dims(img, axis=-1)
        img = tf.image.resize(img, (config.IMG_HEIGHT, config.IMG_WIDTH))
        if img.shape[-1] == 1:  # replicate the single channel into RGB
            img = tf.image.grayscale_to_rgb(img)
        _, mask = inference.predict_mask(model, img)
        return (mask * 255).astype(np.uint8)

    return gr.Interface(
        fn=segment,
        inputs=gr.Image(label="MRI slice"),
        outputs=gr.Image(label="Predicted tumor mask"),
        title="Brain Tumor Segmentation (U-Net)",
        description="Upload an MRI slice to see the predicted tumor mask.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a trained .h5 model.")
    args = parser.parse_args()
    model = inference.load_model(args.model)
    build_interface(model).launch()


if __name__ == "__main__":
    main()
