"""Integration tests for nice_sar.io.products using synthetic NISAR HDF5 files."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from rasterio.transform import Affine

from nice_sar.io.products import (
    get_projection_info,
    read_gcov,
    read_gcov_metadata,
    read_goff,
    read_gslc,
    read_gunw,
    read_identification,
    read_quad_covariances,
    read_rslc,
)


class TestReadGcovMetadata:
    """Tests for read_gcov_metadata."""

    def test_returns_dict(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            meta = read_gcov_metadata(h5)
        assert isinstance(meta, dict)

    def test_expected_keys(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            meta = read_gcov_metadata(h5)
        expected = {"product_type", "start_time", "end_time", "orbit", "track", "frame"}
        assert set(meta.keys()) == expected

    def test_values(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            meta = read_gcov_metadata(h5)
        assert meta["product_type"] == "GCOV"
        assert meta["orbit"] == 1234
        assert meta["track"] == 42
        assert meta["frame"] == 100
        assert "2025-06-01" in meta["start_time"]


class TestGetProjectionInfo:
    """Tests for get_projection_info."""

    def test_returns_tuple(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            result = get_projection_info(h5, frequency="A")
        assert isinstance(result, tuple) and len(result) == 4

    def test_crs(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            crs, _, _, _ = get_projection_info(h5)
        assert isinstance(crs, CRS)
        assert crs.to_epsg() == 32612

    def test_transform(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            _, transform, _, _ = get_projection_info(h5)
        assert isinstance(transform, Affine)
        # x pixel size = 30 m
        assert transform.a == 30.0

    def test_coordinate_shapes(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            _, _, x_coords, y_coords = get_projection_info(h5)
        assert x_coords.shape == (64,)
        assert y_coords.shape == (64,)


class TestReadGcov:
    """Tests for read_gcov."""

    def test_returns_dataarray(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        assert isinstance(da_xr, xr.DataArray)

    def test_shape(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_dask_backed(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        assert isinstance(da_xr.data, da.Array)

    def test_coordinates(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        assert "x" in da_xr.coords
        assert "y" in da_xr.coords
        assert len(da_xr.coords["x"]) == 64

    def test_attrs(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        assert da_xr.attrs["polarization"] == "HH"
        assert da_xr.attrs["frequency"] == "A"
        assert da_xr.attrs["product_type"] == "GCOV"

    def test_hvhv_pol(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HV")
        assert da_xr.shape == (64, 64)
        assert da_xr.attrs["polarization"] == "HV"

    def test_accepts_h5py_file(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            da_xr = read_gcov(h5, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_values_positive(self, synthetic_gcov_path: Path) -> None:
        da_xr = read_gcov(synthetic_gcov_path, polarization="HH")
        vals = da_xr.values
        assert np.all(vals >= 0)


class TestReadQuadCovariances:
    """Tests for read_quad_covariances."""

    def test_returns_dict(self, synthetic_gcov_path: Path) -> None:
        grid = "/science/LSAR/GCOV/grids/frequencyA"
        with h5py.File(synthetic_gcov_path, "r") as h5:
            cov = read_quad_covariances(h5, grid)
        assert isinstance(cov, dict)

    def test_diagonal_terms_real(self, synthetic_gcov_path: Path) -> None:
        grid = "/science/LSAR/GCOV/grids/frequencyA"
        with h5py.File(synthetic_gcov_path, "r") as h5:
            cov = read_quad_covariances(h5, grid)
        assert "HHHH" in cov
        assert cov["HHHH"].dtype == np.float32

    def test_offdiag_terms_complex(self, synthetic_gcov_path: Path) -> None:
        grid = "/science/LSAR/GCOV/grids/frequencyA"
        with h5py.File(synthetic_gcov_path, "r") as h5:
            cov = read_quad_covariances(h5, grid)
        assert "HHHV" in cov
        assert np.iscomplexobj(cov["HHHV"])

    def test_shapes(self, synthetic_gcov_path: Path) -> None:
        grid = "/science/LSAR/GCOV/grids/frequencyA"
        with h5py.File(synthetic_gcov_path, "r") as h5:
            cov = read_quad_covariances(h5, grid)
        for arr in cov.values():
            assert arr.shape == (64, 64)


# ---------------------------------------------------------------------------
# read_identification (shared helper)
# ---------------------------------------------------------------------------


class TestReadIdentification:
    """Tests for the universal read_identification helper."""

    def test_gcov(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            meta = read_identification(h5)
        assert meta["product_type"] == "GCOV"

    def test_gunw(self, synthetic_gunw_path: Path) -> None:
        with h5py.File(synthetic_gunw_path, "r") as h5:
            meta = read_identification(h5)
        assert meta["product_type"] == "GUNW"

    def test_gslc(self, synthetic_gslc_path: Path) -> None:
        with h5py.File(synthetic_gslc_path, "r") as h5:
            meta = read_identification(h5)
        assert meta["product_type"] == "GSLC"

    def test_rslc(self, synthetic_rslc_path: Path) -> None:
        with h5py.File(synthetic_rslc_path, "r") as h5:
            meta = read_identification(h5)
        assert meta["product_type"] == "RSLC"


# ---------------------------------------------------------------------------
# GSLC reader
# ---------------------------------------------------------------------------


class TestReadGslc:
    """Tests for read_gslc."""

    def test_returns_dataarray(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert isinstance(da_xr, xr.DataArray)

    def test_shape(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_complex_dtype(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert np.iscomplexobj(da_xr.values)

    def test_dask_backed(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert isinstance(da_xr.data, da.Array)

    def test_coordinates(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert "x" in da_xr.coords
        assert "y" in da_xr.coords

    def test_attrs(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HH")
        assert da_xr.attrs["product_type"] == "GSLC"
        assert da_xr.attrs["polarization"] == "HH"
        assert da_xr.attrs["units"] == "complex_dn"

    def test_hv_pol(self, synthetic_gslc_path: Path) -> None:
        da_xr = read_gslc(synthetic_gslc_path, polarization="HV")
        assert da_xr.shape == (64, 64)

    def test_accepts_h5py_file(self, synthetic_gslc_path: Path) -> None:
        with h5py.File(synthetic_gslc_path, "r") as h5:
            da_xr = read_gslc(h5, polarization="HH")
        assert da_xr.shape == (64, 64)


# ---------------------------------------------------------------------------
# GUNW reader
# ---------------------------------------------------------------------------


class TestReadGunw:
    """Tests for read_gunw."""

    def test_returns_dataarray(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(synthetic_gunw_path, polarization="HH")
        assert isinstance(da_xr, xr.DataArray)

    def test_shape(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(synthetic_gunw_path, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_default_layer(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(synthetic_gunw_path, polarization="HH")
        assert da_xr.attrs["layer"] == "unwrappedPhase"
        assert da_xr.attrs["units"] == "radians"

    def test_coherence_layer(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(
            synthetic_gunw_path, polarization="HH", layer="coherenceMagnitude"
        )
        assert da_xr.attrs["units"] == "unitless"
        vals = da_xr.values
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_wrapped_interferogram_complex(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(
            synthetic_gunw_path, polarization="HH", layer="wrappedInterferogram"
        )
        assert np.iscomplexobj(da_xr.values)

    def test_connected_components_uint(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(
            synthetic_gunw_path, polarization="HH", layer="connectedComponents"
        )
        assert da_xr.values.dtype == np.uint32

    def test_invalid_layer_raises(self, synthetic_gunw_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown GUNW layer"):
            read_gunw(synthetic_gunw_path, polarization="HH", layer="bogus")

    def test_dask_backed(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(synthetic_gunw_path, polarization="HH")
        assert isinstance(da_xr.data, da.Array)

    def test_coordinates(self, synthetic_gunw_path: Path) -> None:
        da_xr = read_gunw(synthetic_gunw_path, polarization="HH")
        assert "x" in da_xr.coords and "y" in da_xr.coords

    def test_accepts_h5py_file(self, synthetic_gunw_path: Path) -> None:
        with h5py.File(synthetic_gunw_path, "r") as h5:
            da_xr = read_gunw(h5, polarization="HH")
        assert da_xr.shape == (64, 64)


# ---------------------------------------------------------------------------
# GOFF reader
# ---------------------------------------------------------------------------


class TestReadGoff:
    """Tests for read_goff."""

    def test_returns_dataarray(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH")
        assert isinstance(da_xr, xr.DataArray)

    def test_shape(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_default_layer(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH")
        assert da_xr.attrs["layer"] == "alongTrackOffset"
        assert da_xr.attrs["units"] == "pixels"

    def test_slant_range_layer(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(
            synthetic_goff_path, polarization="HH", layer="slantRangeOffset"
        )
        assert da_xr.attrs["layer"] == "slantRangeOffset"

    def test_snr_layer(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH", layer="snr")
        assert da_xr.attrs["units"] == "ratio"
        assert np.all(da_xr.values > 0)

    def test_invalid_layer_raises(self, synthetic_goff_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown GOFF layer"):
            read_goff(synthetic_goff_path, polarization="HH", layer="bogus")

    def test_dask_backed(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH")
        assert isinstance(da_xr.data, da.Array)

    def test_coordinates(self, synthetic_goff_path: Path) -> None:
        da_xr = read_goff(synthetic_goff_path, polarization="HH")
        assert "x" in da_xr.coords and "y" in da_xr.coords

    def test_accepts_h5py_file(self, synthetic_goff_path: Path) -> None:
        with h5py.File(synthetic_goff_path, "r") as h5:
            da_xr = read_goff(h5, polarization="HH")
        assert da_xr.shape == (64, 64)


# ---------------------------------------------------------------------------
# RSLC reader
# ---------------------------------------------------------------------------


class TestReadRslc:
    """Tests for read_rslc."""

    def test_returns_dataarray(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert isinstance(da_xr, xr.DataArray)

    def test_shape(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert da_xr.shape == (64, 64)

    def test_complex_dtype(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert np.iscomplexobj(da_xr.values)

    def test_radar_geometry_dims(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert da_xr.dims == ("azimuth", "range")

    def test_has_slant_range_coord(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert "range" in da_xr.coords
        assert len(da_xr.coords["range"]) == 64

    def test_has_azimuth_coord(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert "azimuth" in da_xr.coords
        assert len(da_xr.coords["azimuth"]) == 64

    def test_no_crs_attr(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert "crs" not in da_xr.attrs

    def test_dask_backed(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HH")
        assert isinstance(da_xr.data, da.Array)

    def test_hv_pol(self, synthetic_rslc_path: Path) -> None:
        da_xr = read_rslc(synthetic_rslc_path, polarization="HV")
        assert da_xr.shape == (64, 64)

    def test_accepts_h5py_file(self, synthetic_rslc_path: Path) -> None:
        with h5py.File(synthetic_rslc_path, "r") as h5:
            da_xr = read_rslc(h5, polarization="HH")
        assert da_xr.shape == (64, 64)
