"""Tests for nice_sar.io.geotiff."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from nice_sar.io.geotiff import export_geotiff, write_rgb_geotiff_uint8, read_band


@pytest.fixture()
def transform():
    """Simple affine transform for testing."""
    from rasterio.transform import Affine
    return Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)


@pytest.fixture()
def crs():
    """Simple CRS for testing."""
    from pyproj import CRS
    return CRS.from_epsg(32611)


class TestExportGeotiff:
    def test_roundtrip(self, tmp_path: Path, transform, crs) -> None:
        data = np.random.default_rng(42).random((64, 64)).astype(np.float32)
        out = tmp_path / "test.tif"
        export_geotiff(data, out, transform, crs, description="test_band")
        assert out.exists()

        # Read back and verify
        read_data, profile = read_band(out)
        assert read_data.shape == (64, 64)
        np.testing.assert_allclose(read_data, data, rtol=1e-5)
        assert profile["crs"].to_epsg() == 32611

    def test_creates_sidecar(self, tmp_path: Path, transform, crs) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        out = tmp_path / "sigma0_test.tif"
        export_geotiff(data, out, transform, crs)
        sidecar = out.with_suffix(".tif.json")
        assert sidecar.exists()

    def test_creates_parent_dirs(self, tmp_path: Path, transform, crs) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        out = tmp_path / "subdir" / "nested" / "test.tif"
        export_geotiff(data, out, transform, crs)
        assert out.exists()


class TestWriteRgbGeotiffUint8:
    def test_float32_input(self, tmp_path: Path, transform, crs) -> None:
        rng = np.random.default_rng(42)
        rgb = rng.random((3, 32, 32)).astype(np.float32)
        out = tmp_path / "rgb_float.tif"
        write_rgb_geotiff_uint8(rgb, out, transform, crs)
        assert out.exists()

        import rasterio
        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.dtypes[0] == "uint8"

    def test_uint8_input_no_double_stretch(self, tmp_path: Path, transform, crs) -> None:
        """uint8 input should be written as-is, not re-stretched."""
        rgb = np.full((3, 16, 16), 128, dtype=np.uint8)
        out = tmp_path / "rgb_u8.tif"
        write_rgb_geotiff_uint8(rgb, out, transform, crs)

        import rasterio
        with rasterio.open(out) as src:
            data = src.read()
            np.testing.assert_array_equal(data, rgb)

    def test_invalid_shape_raises(self, tmp_path: Path, transform, crs) -> None:
        rgb = np.ones((4, 10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected"):
            write_rgb_geotiff_uint8(rgb, tmp_path / "bad.tif", transform, crs)
