"""Experimental Keras Grad-CAM adapter.

The ArabGlyphDiag validation did not establish that Grad-CAM alignment reliably
tracks the empirical secondary-component priors. This module is therefore
provided as an optional exploratory adapter, not as validated structural
reasoning evidence.
"""

from __future__ import annotations

import numpy as np


def gradcam_heatmap(model, image, conv_layer_name: str, class_index: int | None = None) -> np.ndarray:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for arab_glyph_diag.explain.keras. "
            "Install with: pip install 'arabglyphdiag[keras]'"
        ) from exc

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    x = tf.convert_to_tensor(np.asarray(image)[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if isinstance(preds, dict):
            preds = preds.get("letter", next(iter(preds.values())))
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.maximum(heat, 0)
    denom = tf.reduce_max(heat)
    heat = tf.where(denom > 0, heat / denom, heat)
    return heat.numpy().astype(np.float32)
