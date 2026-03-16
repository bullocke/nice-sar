"""Tests for nice_sar.preprocess.filters."""

from __future__ import annotations

import numpy as np

from nice_sar.preprocess.filters import lee_filter, refined_lee_filter


class TestLeeFilter:
    def test_reduces_variance(self) -> None:
        rng = np.random.default_rng(42)
        # Simulate speckled SAR data (exponential distribution)
        data = rng.exponential(0.05, size=(100, 100)).astype(np.float32)
        filtered = lee_filter(data, window_size=5)
        assert filtered.shape == data.shape
        # Filtered data should have lower variance
        assert np.nanvar(filtered) < np.nanvar(data)

    def test_preserves_mean(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(0.05, size=(100, 100)).astype(np.float32)
        filtered = lee_filter(data, window_size=5)
        # Mean should be approximately preserved
        np.testing.assert_allclose(
            np.nanmean(filtered), np.nanmean(data), rtol=0.1
        )

    def test_handles_nan(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(0.05, size=(50, 50)).astype(np.float32)
        data[20:30, 20:30] = np.nan
        filtered = lee_filter(data, window_size=5)
        # NaN regions should stay NaN
        assert np.all(np.isnan(filtered[20:30, 20:30]))
        # Non-NaN regions should have valid values
        assert np.all(np.isfinite(filtered[:15, :15]))

    def test_all_nan(self) -> None:
        data = np.full((10, 10), np.nan, dtype=np.float32)
        filtered = lee_filter(data)
        assert np.all(np.isnan(filtered))


class TestRefinedLeeFilter:
    def test_reduces_variance(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(0.05, size=(100, 100)).astype(np.float32)
        filtered = refined_lee_filter(data, win=7)
        assert filtered.shape == data.shape
        assert np.nanvar(filtered) < np.nanvar(data)

    def test_minimum_window(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(0.05, size=(20, 20)).astype(np.float32)
        # win < 5 should be upgraded to 5
        filtered = refined_lee_filter(data, win=3)
        assert filtered.shape == data.shape

    def test_handles_nan(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(0.05, size=(30, 30)).astype(np.float32)
        data[10:15, 10:15] = np.nan
        filtered = refined_lee_filter(data, win=7)
        assert np.all(np.isnan(filtered[10:15, 10:15]))
