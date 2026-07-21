from __future__ import annotations

import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_snr(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int,
    input_order: str = "HWC",
    **kwargs: object,
) -> float:
    """Calculate reference SNR using the GT signal and prediction error.

    Args:
        img: Predicted image.
        img2: Ground-truth image.
        crop_border: Pixels removed from every image edge.
        input_order: ``HWC`` or ``CHW``.

    Returns:
        ``10 * log10(sum(GT^2) / sum((prediction - GT)^2))``.
    """
    if img.shape != img2.shape:
        raise ValueError(f"Image shapes are different: {img.shape}, {img2.shape}")
    prediction = reorder_image(img, input_order=input_order).astype(np.float64)
    ground_truth = reorder_image(img2, input_order=input_order).astype(np.float64)
    if crop_border:
        prediction = prediction[crop_border:-crop_border, crop_border:-crop_border, ...]
        ground_truth = ground_truth[crop_border:-crop_border, crop_border:-crop_border, ...]

    signal_power = float(np.sum(ground_truth**2))
    error_power = float(np.sum((prediction - ground_truth) ** 2))
    if error_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(signal_power / error_power))
