import numpy as np
import pytest

from basicsr.metrics.snr import calculate_snr


def test_calculate_snr() -> None:
    ground_truth = np.full((4, 4, 1), 2.0)
    prediction = np.full((4, 4, 1), 1.0)
    expected = 10.0 * np.log10(4.0)

    assert calculate_snr(prediction, ground_truth, crop_border=0) == pytest.approx(expected)


def test_calculate_snr_perfect_prediction() -> None:
    image = np.ones((4, 4, 1))
    assert calculate_snr(image, image, crop_border=0) == float("inf")


def test_calculate_snr_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="Image shapes are different"):
        calculate_snr(np.ones((4, 4, 1)), np.ones((3, 4, 1)), crop_border=0)
