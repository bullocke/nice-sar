"""Integration tests for GLCM texture with real NISAR GCOV data.

These tests require the GCOV GeoTIFF raster at
``NISAR_Data/GCOV/GCOV_freqA_HV_HVHV_2026-01-20_subset.tif``.
They are skipped automatically when the file is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "NISAR_Data" / "GCOV"
HV_TIF = DATA_DIR / "GCOV_freqA_HV_HVHV_2026-01-20_subset.tif"

needs_data = pytest.mark.skipif(
    not HV_TIF.exists(),
    reason="GCOV GeoTIFF not found — run notebooks first",
)


@needs_data
class TestGlcmIntegration:
    """Run GLCM on a small crop of real NISAR backscatter."""

    @pytest.fixture(autouse=True)
    def _load_crop(self) -> None:
        """Read a 50×50 crop from the HV band and convert to dB."""
        from nice_sar.io.geotiff import read_band
        from nice_sar.preprocess.calibration import linear_to_db

        arr, _ = read_band(HV_TIF)
        # Take a small crop from the centre of the scene
        r0 = arr.shape[0] // 2
        c0 = arr.shape[1] // 2
        crop = arr[r0 : r0 + 50, c0 : c0 + 50]
        self.crop_db = linear_to_db(crop)

    def test_runs_without_error(self) -> None:
        from nice_sar.preprocess.texture import compute_glcm_texture

        result = compute_glcm_texture(
            self.crop_db, window_size=7, levels=32
        )
        assert result is not None
        assert len(result) == 13

    def test_finite_features(self) -> None:
        from nice_sar.preprocess.texture import compute_glcm_texture

        result = compute_glcm_texture(
            self.crop_db, window_size=7, levels=32
        )
        assert result is not None
        for name, arr in result.items():
            valid = np.isfinite(self.crop_db)
            assert np.any(np.isfinite(arr[valid])), f"{name}: no finite values"

    def test_contrast_range(self) -> None:
        from nice_sar.preprocess.texture import compute_glcm_texture

        result = compute_glcm_texture(
            self.crop_db, window_size=7, levels=32
        )
        assert result is not None
        contrast = result["contrast"]
        finite = contrast[np.isfinite(contrast)]
        assert finite.min() >= 0, "Contrast must be non-negative"

    def test_asm_range(self) -> None:
        from nice_sar.preprocess.texture import compute_glcm_texture

        result = compute_glcm_texture(
            self.crop_db, window_size=7, levels=32
        )
        assert result is not None
        asm = result["asm"]
        finite = asm[np.isfinite(asm)]
        assert finite.min() >= 0 and finite.max() <= 1.0, "ASM must be in [0, 1]"

    def test_rank_vs_glcm_shapes(self) -> None:
        """Both texture methods should return the same spatial shape."""
        from nice_sar.preprocess.texture import (
            compute_glcm_texture,
            compute_rank_texture,
        )

        rank = compute_rank_texture(self.crop_db, window_size=7, levels=32)
        glcm = compute_glcm_texture(self.crop_db, window_size=7, levels=32)
        assert rank is not None and glcm is not None
        for k_r in rank:
            assert rank[k_r].shape == self.crop_db.shape
        for k_g in glcm:
            assert glcm[k_g].shape == self.crop_db.shape
