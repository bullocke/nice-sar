"""Tests for nice_sar.viz.display."""

from __future__ import annotations

import numpy as np

from nice_sar.viz.display import gamma_correct, percentile_stretch, to_uint8


class TestPercentileStretch:
    def test_output_range(self, hh_linear: np.ndarray) -> None:
        result = percentile_stretch(hh_linear)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_percentiles(self, hh_linear: np.ndarray) -> None:
        result = percentile_stretch(hh_linear, p_low=5, p_high=95)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_constant_input(self) -> None:
        arr = np.ones((10, 10), dtype=np.float32) * 5.0
        result = percentile_stretch(arr)
        assert np.isfinite(result).all()


class TestGammaCorrect:
    def test_identity(self) -> None:
        arr = np.linspace(0, 1, 100, dtype=np.float32)
        result = gamma_correct(arr, gamma=1.0)
        np.testing.assert_allclose(result, arr, rtol=1e-6)

    def test_brightening(self) -> None:
        arr = np.array([0.25], dtype=np.float32)
        result = gamma_correct(arr, gamma=0.5)
        assert result[0] > 0.25


class TestToUint8:
    def test_range(self) -> None:
        arr = np.linspace(0, 1, 256, dtype=np.float32)
        result = to_uint8(arr)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255
