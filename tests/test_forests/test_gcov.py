"""Tests for GCOV forest masking methods."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nice_sar.forests.gcov import (
    gcov_hv_threshold,
    gcov_hv_threshold_jaxa_biomass,
    gcov_hv_threshold_ramachandran,
)
from nice_sar.preprocess.calibration import db_to_linear


class TestGcovHvThreshold:
    def test_threshold_on_db_array(self) -> None:
        hv_db = np.array([[-25.0, -19.0], [-13.0, -12.0]], dtype=np.float32)
        mask = gcov_hv_threshold(hv_db, threshold_db=-20.0, units="db")
        expected = np.array([[False, True], [True, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_threshold_on_linear_array(self) -> None:
        hv_db = np.array([[-25.0, -19.0], [-13.0, -12.0]], dtype=np.float32)
        hv_linear = db_to_linear(hv_db)
        mask = gcov_hv_threshold(hv_linear, threshold_db=-20.0, units="linear_power")
        expected = np.array([[False, True], [True, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_preserves_xarray_shape_and_attrs(self) -> None:
        hv = xr.DataArray(
            np.array([[-18.0, -22.0], [-14.0, -11.0]], dtype=np.float32),
            dims=["y", "x"],
            attrs={"units": "db", "crs": "EPSG:32612", "transform": (30, 0, 0, 0, -30, 0)},
        )
        mask = gcov_hv_threshold_jaxa_biomass(hv)
        assert isinstance(mask, xr.DataArray)
        assert mask.shape == hv.shape
        assert mask.attrs["threshold_db"] == -13.0
        assert mask.attrs["mask_method"] == "gcov_hv_threshold"

    def test_ramachandran_preset_uses_minus20(self) -> None:
        hv_db = np.array([[-20.1, -20.0, -19.9]], dtype=np.float32)
        mask = gcov_hv_threshold_ramachandran(hv_db, units="db")
        expected = np.array([[False, True, True]])
        np.testing.assert_array_equal(mask, expected)