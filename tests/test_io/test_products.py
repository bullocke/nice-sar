"""Integration tests for nice_sar.io.products using a synthetic GCOV HDF5 file."""

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
    read_quad_covariances,
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
