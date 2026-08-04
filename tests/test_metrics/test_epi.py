import numpy as np
import pytest

from basicsr.metrics.epi import calculate_epi


def test_calculate_epi_identical() -> None:
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float64).reshape(64, 64, 1)
    image[:, 32:, :] = 1.0
    assert calculate_epi(image, image, crop_border=0) == pytest.approx(1.0, abs=1e-6)


def test_calculate_epi_blur_lowers_score() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.random((64, 64, 1))
    sharp[:, 30:34, :] = 1.0
    kernel = np.ones((5, 5), dtype=np.float64) / 25.0
    from numpy.lib.stride_tricks import sliding_window_view

    padded = np.pad(sharp[..., 0], 2, mode="edge")
    windows = sliding_window_view(padded, (5, 5))
    blur = np.einsum("ij,hwij->hw", kernel, windows)[..., None]
    epi_id = calculate_epi(sharp, sharp, crop_border=0)
    epi_blur = calculate_epi(blur, sharp, crop_border=0)
    assert epi_id > epi_blur


def test_calculate_epi_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="Image shapes are different"):
        calculate_epi(np.ones((4, 4, 1)), np.ones((3, 4, 1)), crop_border=0)
