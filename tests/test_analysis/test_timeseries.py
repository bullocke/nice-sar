"""Tests for nice_sar.analysis.timeseries."""

from __future__ import annotations

import numpy as np

from nice_sar.analysis.timeseries import (
    CUSUMResult,
    HarmonicResult,
    ThresholdResult,
    backscatter_threshold,
    coefficient_of_variation,
    cusum,
    harmonic_fit,
)


class TestCUSUM:
    def test_returns_result(self) -> None:
        rng = np.random.default_rng(60)
        stack = rng.normal(1.0, 0.1, size=(20, 8, 8)).astype(np.float32)
        result = cusum(stack)
        assert isinstance(result, CUSUMResult)

    def test_output_shapes(self) -> None:
        stack = np.ones((10, 5, 5), dtype=np.float32)
        result = cusum(stack)
        assert result.magnitude.shape == (5, 5)
        assert result.change_index.shape == (5, 5)
        assert result.cusum_series.shape == (10, 5, 5)

    def test_detects_step_change(self) -> None:
        # Construct stack with a clear step change at t=10
        stack = np.ones((20, 4, 4), dtype=np.float32)
        stack[10:] = 5.0  # Big jump
        result = cusum(stack, baseline_count=8)
        # Change index should cluster around t=10 or later
        assert np.all(result.change_index >= 8)

    def test_threshold_masks(self) -> None:
        stack = np.ones((10, 4, 4), dtype=np.float32)
        result = cusum(stack, threshold=1e6)
        # Constant input → small CUSUM → should be masked to -1
        assert np.all(result.change_index == -1)


class TestCoefficientOfVariation:
    def test_output_shape(self) -> None:
        stack = np.ones((10, 8, 8), dtype=np.float32)
        cov = coefficient_of_variation(stack)
        assert cov.shape == (8, 8)

    def test_constant_stack_zero_cov(self) -> None:
        stack = np.full((10, 4, 4), 5.0, dtype=np.float32)
        cov = coefficient_of_variation(stack)
        np.testing.assert_allclose(cov, 0.0, atol=1e-6)

    def test_variable_stack_positive_cov(self) -> None:
        rng = np.random.default_rng(61)
        stack = rng.exponential(1.0, size=(30, 8, 8)).astype(np.float32)
        cov = coefficient_of_variation(stack)
        assert np.all(cov > 0)

    def test_dtype(self) -> None:
        stack = np.ones((5, 3, 3), dtype=np.float64)
        cov = coefficient_of_variation(stack)
        assert cov.dtype == np.float32


class TestBackscatterThreshold:
    def test_returns_result(self) -> None:
        stack = np.ones((5, 4, 4), dtype=np.float32) * 0.5
        result = backscatter_threshold(stack, low=0.3, high=0.7)
        assert isinstance(result, ThresholdResult)

    def test_all_in_range(self) -> None:
        stack = np.ones((5, 4, 4), dtype=np.float32) * 0.5
        result = backscatter_threshold(stack, low=0.0, high=1.0)
        assert np.all(result.mask)
        np.testing.assert_allclose(result.fraction, 1.0)

    def test_none_in_range(self) -> None:
        stack = np.ones((5, 4, 4), dtype=np.float32) * 10.0
        result = backscatter_threshold(stack, low=0.0, high=1.0, min_fraction=0.5)
        assert not np.any(result.mask)

    def test_min_fraction(self) -> None:
        stack = np.zeros((10, 3, 3), dtype=np.float32)
        stack[:3] = 1.0  # 30% of time in range
        result = backscatter_threshold(stack, low=0.5, high=1.5, min_fraction=0.5)
        assert not np.any(result.mask)
        result2 = backscatter_threshold(stack, low=0.5, high=1.5, min_fraction=0.2)
        assert np.all(result2.mask)


class TestHarmonicFit:
    def test_returns_result(self) -> None:
        rng = np.random.default_rng(62)
        T = 24
        times = np.linspace(0, 365, T)
        stack = rng.normal(1.0, 0.1, size=(T, 4, 4)).astype(np.float32)
        result = harmonic_fit(stack, times)
        assert isinstance(result, HarmonicResult)

    def test_output_shapes(self) -> None:
        T = 12
        times = np.linspace(0, 365, T)
        stack = np.ones((T, 5, 5), dtype=np.float32)
        result = harmonic_fit(stack, times)
        assert result.intercept.shape == (5, 5)
        assert result.amplitude.shape == (5, 5)
        assert result.residuals.shape == (T, 5, 5)
        assert result.rmse.shape == (5, 5)

    def test_recovers_known_signal(self) -> None:
        T = 48
        times = np.linspace(0, 365.25 * 2, T)
        omega = 2 * np.pi / 365.25
        # Known signal: 1.0 + 0.5*cos(ωt)
        signal = 1.0 + 0.5 * np.cos(omega * times)
        stack = np.tile(signal[:, None, None], (1, 3, 3)).astype(np.float32)
        result = harmonic_fit(stack, times, period=365.25)
        np.testing.assert_allclose(result.intercept[0, 0], 1.0, atol=0.05)
        np.testing.assert_allclose(result.amplitude[0, 0], 0.5, atol=0.05)

    def test_constant_input_small_amplitude(self) -> None:
        T = 12
        times = np.linspace(0, 365, T)
        stack = np.full((T, 3, 3), 2.0, dtype=np.float32)
        result = harmonic_fit(stack, times)
        np.testing.assert_allclose(result.amplitude, 0.0, atol=1e-4)
        np.testing.assert_allclose(result.intercept, 2.0, atol=1e-4)
