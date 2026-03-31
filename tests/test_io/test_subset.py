"""Tests for nice_sar.io.bbox_parser and nice_sar.io.subset."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from pyproj import CRS

from nice_sar.io.bbox_parser import parse_bbox, validate_bbox
from nice_sar.io.subset import bbox_to_pixel_slices, estimate_subset_size, subset_product

# -----------------------------------------------------------------------
# Shared constants — match the synthetic GCOV fixture (EPSG:32612)
# -----------------------------------------------------------------------
_FULL_BBOX = (-111.888632, 40.765539, -111.866463, 40.782734)
_SUBSET_BBOX = (-111.882999, 40.769636, -111.871739, 40.778370)

_GUYANA_GEOJSON = (
    '{"type":"Polygon","coordinates":[[[-58.236822602728694,4.568883563565226],'
    "[-58.236822602728694,4.398774446109351],"
    "[-58.06241464374432,4.398774446109351],"
    '[-58.06241464374432,4.568883563565226]]],"geodesic":false,"evenOdd":true}'
)


# =======================================================================
# bbox_parser tests
# =======================================================================


class TestValidateBbox:
    def test_valid(self) -> None:
        bbox = (-112.0, 40.5, -111.7, 40.9)
        assert validate_bbox(bbox) == bbox

    def test_south_gte_north(self) -> None:
        with pytest.raises(ValueError, match="South"):
            validate_bbox((-112.0, 41.0, -111.7, 40.9))

    def test_west_gte_east(self) -> None:
        with pytest.raises(ValueError, match="West"):
            validate_bbox((-111.0, 40.5, -112.0, 40.9))

    def test_lon_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Longitude"):
            validate_bbox((-200.0, 40.5, -111.7, 40.9))

    def test_lat_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Latitude"):
            validate_bbox((-112.0, -100.0, -111.7, 40.9))


class TestParseBbox:
    def test_tuple(self) -> None:
        bbox = (-112.0, 40.5, -111.7, 40.9)
        assert parse_bbox(bbox) == bbox

    def test_list(self) -> None:
        assert parse_bbox([-112.0, 40.5, -111.7, 40.9]) == (-112.0, 40.5, -111.7, 40.9)

    def test_csv_string(self) -> None:
        assert parse_bbox("-112.0, 40.5, -111.7, 40.9") == (-112.0, 40.5, -111.7, 40.9)

    def test_geojson_string(self) -> None:
        bbox = parse_bbox(_GUYANA_GEOJSON)
        west, south, east, north = bbox
        assert west == pytest.approx(-58.2368, abs=0.001)
        assert south == pytest.approx(4.3988, abs=0.001)
        assert east == pytest.approx(-58.0624, abs=0.001)
        assert north == pytest.approx(4.5689, abs=0.001)

    def test_geojson_dict(self) -> None:
        geom = json.loads(_GUYANA_GEOJSON)
        bbox = parse_bbox(geom)
        assert len(bbox) == 4
        assert bbox[0] < bbox[2]  # west < east
        assert bbox[1] < bbox[3]  # south < north

    def test_geojson_feature(self) -> None:
        geom = json.loads(_GUYANA_GEOJSON)
        feature = {"type": "Feature", "geometry": geom, "properties": {}}
        bbox = parse_bbox(feature)
        assert bbox[0] == pytest.approx(-58.2368, abs=0.001)

    def test_geojson_file(self, tmp_path: Path) -> None:
        geojson_path = tmp_path / "aoi.geojson"
        geojson_path.write_text(_GUYANA_GEOJSON)
        bbox = parse_bbox(str(geojson_path))
        assert bbox[0] == pytest.approx(-58.2368, abs=0.001)

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_bbox("not_a_bbox")

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_bbox("/nonexistent/file.geojson")

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            parse_bbox(42)


# =======================================================================
# bbox_to_pixel_slices tests
# =======================================================================


class TestBboxToPixelSlices:
    def test_full_bbox(self, synthetic_gcov_path: Path) -> None:
        """Full bbox should return all or nearly all pixels."""
        with h5py.File(synthetic_gcov_path, "r") as h5:
            grid = "/science/LSAR/GCOV/grids/frequencyA"
            x = h5[f"{grid}/xCoordinates"][:]
            y = h5[f"{grid}/yCoordinates"][:]
            crs = CRS.from_epsg(32612)

        row_sl, col_sl, sub_x, sub_y = bbox_to_pixel_slices(x, y, crs, _FULL_BBOX)
        # Should cover most of the 64×64 grid
        assert len(sub_x) >= 50
        assert len(sub_y) >= 50

    def test_subset_bbox(self, synthetic_gcov_path: Path) -> None:
        """Subset bbox should return roughly 32×32 pixels."""
        with h5py.File(synthetic_gcov_path, "r") as h5:
            grid = "/science/LSAR/GCOV/grids/frequencyA"
            x = h5[f"{grid}/xCoordinates"][:]
            y = h5[f"{grid}/yCoordinates"][:]
            crs = CRS.from_epsg(32612)

        row_sl, col_sl, sub_x, sub_y = bbox_to_pixel_slices(x, y, crs, _SUBSET_BBOX)
        assert 20 <= len(sub_x) <= 45
        assert 20 <= len(sub_y) <= 45

    def test_no_overlap(self, synthetic_gcov_path: Path) -> None:
        """Bbox far from the product extent should raise ValueError."""
        with h5py.File(synthetic_gcov_path, "r") as h5:
            grid = "/science/LSAR/GCOV/grids/frequencyA"
            x = h5[f"{grid}/xCoordinates"][:]
            y = h5[f"{grid}/yCoordinates"][:]
            crs = CRS.from_epsg(32612)

        with pytest.raises(ValueError, match="does not overlap"):
            bbox_to_pixel_slices(x, y, crs, (-58.0, 4.0, -57.0, 5.0))


# =======================================================================
# estimate_subset_size tests
# =======================================================================


class TestEstimateSubsetSize:
    def test_known_size(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            total, human = estimate_subset_size(
                h5,
                "GCOV",
                "A",
                ["HH"],
                slice(0, 32),
                slice(0, 32),
            )
        # 32 * 32 * 4 bytes (float32) = 4096
        assert total == 4096
        assert "KB" in human or "B" in human

    def test_multiple_polarizations(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            total, _ = estimate_subset_size(
                h5,
                "GCOV",
                "A",
                ["HH", "HV"],
                slice(0, 32),
                slice(0, 32),
            )
        assert total == 4096 * 2


# =======================================================================
# subset_product round-trip tests
# =======================================================================


class TestSubsetProduct:
    def test_subset_gcov(self, synthetic_gcov_path: Path, tmp_path: Path) -> None:
        """Round-trip: synthetic GCOV → subset → GeoTIFF with correct shape/CRS."""
        outputs = subset_product(
            source=synthetic_gcov_path,
            product="GCOV",
            bbox=_SUBSET_BBOX,
            frequency="A",
            polarizations=["HH"],
            output_dir=tmp_path / "gcov_out",
            confirm=False,
        )
        assert len(outputs) == 1
        assert outputs[0].suffix == ".tif"
        assert outputs[0].exists()

        with rasterio.open(outputs[0]) as ds:
            assert ds.crs is not None
            assert ds.crs.to_epsg() == 32612
            # Should be a subset, not the full 64×64
            assert ds.width < 64 or ds.height < 64
            assert ds.width >= 20
            assert ds.height >= 20
            data = ds.read(1)
            assert data.dtype == np.float32
            assert np.any(data > 0)

    def test_subset_gcov_auto_polarizations(
        self, synthetic_gcov_path: Path, tmp_path: Path
    ) -> None:
        """Auto-detect polarizations when not specified."""
        outputs = subset_product(
            source=synthetic_gcov_path,
            product="GCOV",
            bbox=_SUBSET_BBOX,
            output_dir=tmp_path / "gcov_auto",
            confirm=False,
        )
        # Fixture has HH and HV
        assert len(outputs) == 2

    def test_subset_gunw(self, synthetic_gunw_path: Path, tmp_path: Path) -> None:
        """Subset GUNW with unwrappedPhase layer."""
        outputs = subset_product(
            source=synthetic_gunw_path,
            product="GUNW",
            bbox=_FULL_BBOX,
            frequency="A",
            polarizations=["HH"],
            layers=["unwrappedPhase"],
            output_dir=tmp_path / "gunw_out",
            confirm=False,
        )
        assert len(outputs) == 1
        assert outputs[0].exists()

    def test_subset_gslc(self, synthetic_gslc_path: Path, tmp_path: Path) -> None:
        """Subset GSLC — complex data should be exported as amplitude."""
        outputs = subset_product(
            source=synthetic_gslc_path,
            product="GSLC",
            bbox=_FULL_BBOX,
            frequency="A",
            polarizations=["HH"],
            output_dir=tmp_path / "gslc_out",
            confirm=False,
        )
        assert len(outputs) == 1
        with rasterio.open(outputs[0]) as ds:
            data = ds.read(1)
            assert data.dtype == np.float32  # Amplitude, not complex

    def test_subset_goff(self, synthetic_goff_path: Path, tmp_path: Path) -> None:
        """Subset GOFF alongTrackOffset layer."""
        outputs = subset_product(
            source=synthetic_goff_path,
            product="GOFF",
            bbox=_FULL_BBOX,
            frequency="A",
            polarizations=["HH"],
            layers=["alongTrackOffset"],
            output_dir=tmp_path / "goff_out",
            confirm=False,
        )
        assert len(outputs) == 1

    def test_unsupported_product(self, synthetic_gcov_path: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported product"):
            subset_product(
                source=synthetic_gcov_path,
                product="RSLC",
                bbox=_SUBSET_BBOX,
                output_dir=tmp_path,
                confirm=False,
            )

    def test_no_overlap(self, synthetic_gcov_path: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not overlap"):
            subset_product(
                source=synthetic_gcov_path,
                product="GCOV",
                bbox=(-58.0, 4.0, -57.0, 5.0),
                output_dir=tmp_path,
                confirm=False,
            )
