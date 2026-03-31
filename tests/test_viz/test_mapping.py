"""Tests for nice_sar.viz.mapping."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS
from rasterio.transform import Affine

from nice_sar.viz.mapping import _apply_colormap, _to_latlon_bounds


class TestToLatlonBounds:
    def test_returns_nested_list(self) -> None:
        crs = CRS.from_epsg(32612)
        transform = Affine(30.0, 0.0, 425_000.0, 0.0, -30.0, 4_515_000.0)
        bounds = _to_latlon_bounds(crs, transform, height=64, width=64)
        assert len(bounds) == 2
        assert len(bounds[0]) == 2
        assert len(bounds[1]) == 2

    def test_south_lt_north(self) -> None:
        crs = CRS.from_epsg(32612)
        transform = Affine(30.0, 0.0, 425_000.0, 0.0, -30.0, 4_515_000.0)
        bounds = _to_latlon_bounds(crs, transform, height=64, width=64)
        assert bounds[0][0] < bounds[1][0]  # south < north

    def test_accepts_epsg_int(self) -> None:
        transform = Affine(30.0, 0.0, 425_000.0, 0.0, -30.0, 4_515_000.0)
        bounds = _to_latlon_bounds(32612, transform, height=10, width=10)
        assert bounds[0][0] < bounds[1][0]


class TestApplyColormap:
    def test_output_shape(self) -> None:
        data = np.random.default_rng(70).random((10, 10)).astype(np.float32)
        rgba = _apply_colormap(data)
        assert rgba.shape == (10, 10, 4)
        assert rgba.dtype == np.uint8

    def test_nan_transparent(self) -> None:
        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = np.nan
        rgba = _apply_colormap(data, nodata_alpha=True)
        assert rgba[2, 2, 3] == 0  # alpha = 0 for NaN

    def test_vmin_vmax(self) -> None:
        data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        rgba = _apply_colormap(data, vmin=0.0, vmax=1.0, cmap="gray")
        # With gray colormap, 0→black, 1→white
        assert rgba[0, 0, 0] < rgba[0, 2, 0]  # first pixel darker than last


class TestOverlayRaster:
    """Integration test: only runs if folium and Pillow are available."""

    @pytest.fixture(autouse=True)
    def _check_deps(self) -> None:
        pytest.importorskip("folium")
        pytest.importorskip("PIL")

    def test_creates_map(self) -> None:
        from nice_sar.viz.mapping import overlay_raster

        data = np.random.default_rng(71).random((32, 32)).astype(np.float32)
        crs = CRS.from_epsg(32612)
        transform = Affine(80.0, 0.0, 425_000.0, 0.0, -80.0, 4_515_000.0)
        m = overlay_raster(data, crs, transform)
        import folium

        assert isinstance(m, folium.Map)


class TestOverlayRGB:
    """Integration test with folium."""

    @pytest.fixture(autouse=True)
    def _check_deps(self) -> None:
        pytest.importorskip("folium")
        pytest.importorskip("PIL")

    def test_creates_map(self) -> None:
        from nice_sar.viz.mapping import overlay_rgb

        rgb = np.random.default_rng(72).integers(0, 255, (3, 32, 32), dtype=np.uint8)
        crs = CRS.from_epsg(32612)
        transform = Affine(80.0, 0.0, 425_000.0, 0.0, -80.0, 4_515_000.0)
        m = overlay_rgb(rgb, crs, transform)
        import folium

        assert isinstance(m, folium.Map)

    def test_hwc_format(self) -> None:
        from nice_sar.viz.mapping import overlay_rgb

        rgb = np.random.default_rng(73).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        crs = CRS.from_epsg(32612)
        transform = Affine(80.0, 0.0, 425_000.0, 0.0, -80.0, 4_515_000.0)
        m = overlay_rgb(rgb, crs, transform)
        import folium

        assert isinstance(m, folium.Map)
