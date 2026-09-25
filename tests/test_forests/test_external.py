"""Tests for external forest mask raster support."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nice_sar.forests import generate_forest_mask, load_external_forest_mask
from nice_sar.io.geotiff import export_geotiff
from nice_sar.io.products import read_gcov


def _make_external_mask_raster(output_path: Path, target_path: Path) -> Path:
    target = read_gcov(target_path, frequency="A", polarization="HV")
    from pyproj import CRS
    from rasterio.transform import Affine

    target_crs = CRS.from_user_input(target.attrs["crs"])
    target_transform = Affine(*target.attrs["transform"])
    src_transform = Affine(
        target_transform.a * 2,
        target_transform.b,
        target_transform.c,
        target_transform.d,
        target_transform.e * 2,
        target_transform.f,
    )
    data = np.zeros((32, 32), dtype=np.float32)
    data[:, :16] = 1.0
    export_geotiff(data, output_path, src_transform, target_crs, description="external_mask")
    return output_path


class TestExternalMaskSupport:
    def test_aligns_external_raster_to_target_grid(
        self,
        synthetic_gcov_path: Path,
        tmp_path: Path,
    ) -> None:
        mask_path = _make_external_mask_raster(tmp_path / "external_mask.tif", synthetic_gcov_path)
        result = load_external_forest_mask(mask_path, target=synthetic_gcov_path, threshold=0.5)
        assert result.mask.shape == (64, 64)
        assert result.metadata["target_shape"] == (64, 64)
        left_fraction = float(np.mean(result.mask.values[:, :32]))
        right_fraction = float(np.mean(result.mask.values[:, 32:]))
        assert left_fraction > 0.8
        assert right_fraction < 0.2

    def test_generate_forest_mask_external_method(
        self,
        synthetic_gcov_path: Path,
        tmp_path: Path,
    ) -> None:
        mask_path = _make_external_mask_raster(tmp_path / "external_mask_generate.tif", synthetic_gcov_path)
        result = generate_forest_mask(
            mask_path,
            method="external_raster",
            target=synthetic_gcov_path,
            threshold=0.5,
        )
        assert result.method == "external_raster"
        assert result.mask.shape == (64, 64)