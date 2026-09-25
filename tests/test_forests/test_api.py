"""Tests for the forests high-level API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nice_sar.forests import generate_forest_mask, list_forest_mask_methods


class TestListForestMaskMethods:
    def test_lists_implemented_and_planned_methods(self) -> None:
        methods = list_forest_mask_methods()
        names = {item["name"] for item in methods}
        assert "gcov_hv_threshold" in names
        assert "external_raster" in names
        assert "gcov_dprvi_glcm_texture" in names

    def test_can_filter_unimplemented_methods(self) -> None:
        methods = list_forest_mask_methods(include_unimplemented=False)
        assert all(item["implemented"] for item in methods)


class TestGenerateForestMask:
    def test_generates_mask_from_array(self) -> None:
        hv_db = np.array([[-21.0, -18.0], [-14.0, -24.0]], dtype=np.float32)
        result = generate_forest_mask(hv_db, method="gcov_hv_threshold", units="db")
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(result.mask, expected)
        assert result.threshold == -20.0
        assert result.method == "gcov_hv_threshold"

    def test_generates_mask_from_dataarray(self) -> None:
        hv = xr.DataArray(
            np.array([[-21.0, -18.0], [-14.0, -24.0]], dtype=np.float32),
            dims=["y", "x"],
            attrs={"units": "db", "crs": "EPSG:32612", "transform": (30, 0, 0, 0, -30, 0)},
        )
        result = generate_forest_mask(hv, method="gcov_hv_threshold_jaxa_biomass")
        assert hasattr(result.mask, "attrs")
        assert result.threshold == -13.0

    def test_generates_mask_from_gcov_source(self, synthetic_gcov_path: Path) -> None:
        result = generate_forest_mask(
            synthetic_gcov_path,
            method="gcov_hv_threshold_ramachandran",
            frequency="A",
            polarization="HV",
        )
        assert result.source_kind == "gcov"
        assert result.threshold == -20.0
        assert result.mask.shape == result.confidence.shape

    def test_raises_for_unimplemented_method(self) -> None:
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            generate_forest_mask(np.ones((4, 4), dtype=np.float32), method="gcov_dprvi_glcm_texture")