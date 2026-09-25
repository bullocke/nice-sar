"""Tests for nice_sar.preprocess.texture."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.feature import graycomatrix, graycoprops

from nice_sar.preprocess.texture import (
    HARALICK_FEATURES,
    _haralick_from_glcm,
    compute_glcm_texture,
    compute_local_contrast_homogeneity,
    compute_rank_texture,
)


class TestComputeRankTexture:
    """Tests for compute_rank_texture()."""

    def test_output_keys(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 50)).astype(np.float32)
        result = compute_rank_texture(data, window_size=5, levels=16)
        assert result is not None
        assert set(result.keys()) == {"entropy", "mean", "variance", "range"}

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.standard_normal((40, 60)).astype(np.float32)
        result = compute_rank_texture(data, window_size=5, levels=16)
        assert result is not None
        for key, arr in result.items():
            assert arr.shape == (40, 60), f"{key} shape mismatch"

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((30, 30)).astype(np.float32)
        result = compute_rank_texture(data, window_size=5, levels=16)
        assert result is not None
        for arr in result.values():
            assert arr.dtype == np.float32

    def test_all_nan_returns_none(self) -> None:
        data = np.full((20, 20), np.nan, dtype=np.float32)
        result = compute_rank_texture(data, window_size=5, levels=16)
        assert result is None

    def test_nan_pixels_preserved(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.standard_normal((30, 30)).astype(np.float32)
        data[10, 15] = np.nan
        result = compute_rank_texture(data, window_size=5, levels=16)
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


class TestComputeGlcmTexture:
    """Tests for compute_glcm_texture() — true GLCM Haralick features."""

    def test_output_keys(self) -> None:
        rng = np.random.default_rng(100)
        data = rng.standard_normal((25, 25)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        assert set(result.keys()) == set(HARALICK_FEATURES)

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(101)
        data = rng.standard_normal((20, 30)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for key, arr in result.items():
            assert arr.shape == (20, 30), f"{key} shape mismatch"

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(102)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for arr in result.values():
            assert arr.dtype == np.float32

    def test_all_nan_returns_none(self) -> None:
        data = np.full((15, 15), np.nan, dtype=np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is None

    def test_nan_pixels_preserved(self) -> None:
        rng = np.random.default_rng(103)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        data[8, 8] = np.nan
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        for arr in result.values():
            assert np.isnan(arr[8, 8])

    def test_uniform_region_low_contrast(self) -> None:
        """A constant patch should yield zero contrast and high ASM."""
        data = np.full((25, 25), -15.0, dtype=np.float32)
        result = compute_glcm_texture(data, window_size=5, levels=16)
        assert result is not None
        # Center pixel avoids border effects
        assert result["contrast"][12, 12] == pytest.approx(0.0, abs=1e-5)
        assert result["asm"][12, 12] == pytest.approx(1.0, abs=1e-5)
        assert result["homogeneity"][12, 12] == pytest.approx(1.0, abs=1e-5)

    def test_checkerboard_high_contrast(self) -> None:
        """Alternating pattern should produce high contrast."""
        data = np.zeros((30, 30), dtype=np.float32)
        data[::2, ::2] = 1.0
        data[1::2, 1::2] = 1.0
        result = compute_glcm_texture(data, window_size=7, levels=16)
        assert result is not None
        assert result["contrast"][15, 15] > 0

    def test_feature_subset(self) -> None:
        rng = np.random.default_rng(104)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        result = compute_glcm_texture(
            data, window_size=5, levels=16, features=["contrast", "homogeneity"]
        )
        assert result is not None
        assert set(result.keys()) == {"contrast", "homogeneity"}

    def test_unknown_feature_raises(self) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown Haralick"):
            compute_glcm_texture(data, window_size=5, features=["fake_feature"])

    def test_custom_distances_angles(self) -> None:
        rng = np.random.default_rng(105)
        data = rng.standard_normal((25, 25)).astype(np.float32)
        result = compute_glcm_texture(
            data,
            window_size=5,
            levels=16,
            distances=[1, 2],
            angles=[0, np.pi / 2],
        )
        assert result is not None
        assert set(result.keys()) == set(HARALICK_FEATURES)

    def test_validates_against_skimage(self) -> None:
        """Verify _haralick_from_glcm matches skimage for shared features."""
        rng = np.random.default_rng(200)
        patch = rng.integers(0, 16, size=(15, 15), dtype=np.uint8)

        glcm = graycomatrix(
            patch,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True,
        )
        p = glcm[:, :, 0, 0].astype(np.float64)
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum

        feats = _haralick_from_glcm(p)

        # Compare shared features
        sk_contrast = graycoprops(glcm, "contrast")[0, 0]
        sk_homog = graycoprops(glcm, "homogeneity")[0, 0]
        sk_asm = graycoprops(glcm, "ASM")[0, 0]
        sk_corr = graycoprops(glcm, "correlation")[0, 0]
        sk_energy = graycoprops(glcm, "energy")[0, 0]

        assert feats[HARALICK_FEATURES.index("contrast")] == pytest.approx(sk_contrast, rel=1e-4)
        assert feats[HARALICK_FEATURES.index("homogeneity")] == pytest.approx(sk_homog, rel=1e-4)
        assert feats[HARALICK_FEATURES.index("asm")] == pytest.approx(sk_asm, rel=1e-4)
        assert feats[HARALICK_FEATURES.index("correlation")] == pytest.approx(sk_corr, rel=1e-4)
        # Energy = sqrt(ASM) — verify consistency
        assert np.sqrt(feats[HARALICK_FEATURES.index("asm")]) == pytest.approx(sk_energy, rel=1e-4)


def test_rank_variance_no_uint8_overflow() -> None:
    """Variance stays correct for levels > 16 (squared levels exceed 255)."""
    from nice_sar.preprocess.texture import compute_rank_texture

    img = np.tile(np.array([0.0, 1.0]), (40, 20))  # alternating columns: max variance
    out = compute_rank_texture(img, window_size=5, levels=32)
    assert out is not None
    centre = out["variance"][20, 20]
    assert centre == pytest.approx(31.0**2 / 4, rel=0.15)
    assert np.all(out["variance"] >= 0)
