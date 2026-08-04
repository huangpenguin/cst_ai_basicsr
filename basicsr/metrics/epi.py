from __future__ import annotations

import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D float image."""
    try:
        import cv2

        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return np.hypot(gx, gy)
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view

        kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
        ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64)
        padded = np.pad(gray.astype(np.float64), 1, mode="edge")
        windows = sliding_window_view(padded, (3, 3))
        gx = np.einsum("ij,hwij->hw", kx, windows)
        gy = np.einsum("ij,hwij->hw", ky, windows)
        return np.hypot(gx, gy)


@METRIC_REGISTRY.register()
def calculate_epi(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 0,
    input_order: str = "HWC",
    **kwargs: object,
) -> float:
    """Calculate Edge Preservation Index (EPI) via Sobel-magnitude correlation.

    Args:
        img: Predicted image (HWC/CHW), same scale as ``img2``.
        img2: Ground-truth image.
        crop_border: Pixels removed from every edge before scoring.
        input_order: ``HWC`` or ``CHW``.

    Returns:
        Pearson correlation in ``[-1, 1]`` between Sobel magnitudes of GT and
        prediction. Values near ``1`` mean edges (e.g. metal / electrode) align;
        much lower means the model likely blurred structural boundaries.
    """
    if img.shape != img2.shape:
        raise ValueError(f"Image shapes are different: {img.shape}, {img2.shape}")
    prediction = reorder_image(img, input_order=input_order).astype(np.float64)
    ground_truth = reorder_image(img2, input_order=input_order).astype(np.float64)
    if crop_border:
        prediction = prediction[crop_border:-crop_border, crop_border:-crop_border, ...]
        ground_truth = ground_truth[crop_border:-crop_border, crop_border:-crop_border, ...]

    pred_gray = prediction.mean(axis=2) if prediction.ndim == 3 else prediction
    gt_gray = ground_truth.mean(axis=2) if ground_truth.ndim == 3 else ground_truth
    e_pred = _sobel_magnitude(pred_gray).ravel()
    e_gt = _sobel_magnitude(gt_gray).ravel()
    if e_pred.size < 2:
        return float("nan")
    pred_c = e_pred - e_pred.mean()
    gt_c = e_gt - e_gt.mean()
    denom = float(np.sqrt(np.sum(pred_c**2) * np.sum(gt_c**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(pred_c * gt_c) / denom)
