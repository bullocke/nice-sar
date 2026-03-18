"""Tests for nice_sar.preprocess.texture."""

from __future__ import annotations

import numpy as np

from nice_sar.preprocess.texture import (
    compute_glcm_texture,
    compute_local_contrast_homogeneity,
)


class TestComputeGlcmTexture:
    """Tests for compute_glcm_texture()."""

    def test_output_keys(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 50)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        assert set(result.keys()) == {"entropy", "mean", "variance", "range"}

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.standard_normal((40, 60)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for key, arr in result.items():
            assert arr.shape == (40, 60), f"{key} shape mismatch"

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((30, 30)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for arr in result.values():
            assert arr.dtype == np.float32

    def test_all_nan_returns_none(self) -> None:
        data = np.full((20, 20), np.nan, dtype=np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is None

    def test_nan_pixels_preserved(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.standard_normal((30, 30)).astype(np.float32)
        data[10, 15] = np.nan
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for arr in result.values():
            assert np.isnan(arr[10, 15])


class TestComputeLocalContrastHomogeneity:
    """Tests for compute_local_contrast_homogeneity()."""

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(10)
        data = rng.standard_normal((30, 30)).astype(np.float32)
        contrast, homogeneity = compute_local_contrast_homogeneity(data, window_size=5)
        assert contrast.shape == (30, 30)
        assert homogeneity.shape == (30, 30)

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(11)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        contrast, homogeneity = compute_local_contrast_homogeneity(data, window_size=5)
        assert contrast.dtype == np.float32
        assert homogeneity.dtype == np.float32

    def test_nan_pixels_propagate(self) -> None:
        rng = np.random.default_rng(12)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        data[5, 5] = np.nan
        contrast, homogeneity = compute_local_contrast_homogeneity(data, window_size=5)
        assert np.isnan(contrast[5, 5])
        assert np.isnan(homogeneity[5, 5])
