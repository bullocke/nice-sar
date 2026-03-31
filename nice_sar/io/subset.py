"""Spatial subset and download for NISAR geocoded products.

Stream only the pixels within a bounding box from remote NISAR HDF5 files
(HTTPS or S3), then export each band as a GeoTIFF.  Supports GCOV, GSLC,
GUNW, and GOFF products.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import fsspec
import h5py
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine

from nice_sar._types import BBox, PathType
from nice_sar.io.bbox_parser import parse_bbox, validate_bbox
from nice_sar.io.geotiff import export_geotiff
from nice_sar.io.hdf5 import open_nisar
from nice_sar.io.products import get_projection_info_l2, read_identification

logger = logging.getLogger(__name__)

# Supported geocoded product types
_SUPPORTED_PRODUCTS = {"GCOV", "GSLC", "GUNW", "GOFF"}

# GCOV short polarization → dataset name mapping
_GCOV_POL_MAP: dict[str, str] = {
    "HH": "HHHH",
    "HV": "HVHV",
    "VV": "VVVV",
    "VH": "VHVH",
}

# GUNW known layers and their dtypes
_GUNW_LAYER_DTYPES: dict[str, type] = {
    "unwrappedPhase": np.float32,
    "coherenceMagnitude": np.float32,
    "wrappedInterferogram": np.complex64,
    "connectedComponents": np.uint32,
    "ionospherePhaseScreen": np.float32,
    "ionospherePhaseScreenUncertainty": np.float32,
}

# GOFF known layers
_GOFF_LAYERS = {"alongTrackOffset", "slantRangeOffset", "snr"}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def bbox_to_pixel_slices(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    crs: CRS,
    bbox: BBox,
) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    """Convert a WGS84 bounding box to row/column slices in product coordinates.

    Args:
        x_coords: 1-D array of easting coordinates from the product.
        y_coords: 1-D array of northing coordinates from the product.
        crs: Product CRS (typically UTM).
        bbox: Bounding box as (west, south, east, north) in WGS84 degrees.

    Returns:
        Tuple of (row_slice, col_slice, subset_x_coords, subset_y_coords).

    Raises:
        ValueError: If the bbox does not overlap the product extent.
    """
    bbox = validate_bbox(bbox)
    west, south, east, north = bbox

    # Reproject bbox corners from WGS84 to product CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_min, y_min = transformer.transform(west, south)
    x_max, y_max = transformer.transform(east, north)

    # Handle descending y-coordinates (north-up grids have y[0] > y[-1])
    y_ascending = y_coords[-1] > y_coords[0]
    if y_ascending:
        row_start = int(np.searchsorted(y_coords, y_min, side="left"))
        row_stop = int(np.searchsorted(y_coords, y_max, side="right"))
    else:
        # Descending: flip to ascending, find indices, then convert back
        y_flipped = y_coords[::-1]
        idx_lo = int(np.searchsorted(y_flipped, y_min, side="left"))
        idx_hi = int(np.searchsorted(y_flipped, y_max, side="right"))
        n = len(y_coords)
        row_start = n - idx_hi
        row_stop = n - idx_lo

    # x-coordinates are always ascending (easting)
    col_start = int(np.searchsorted(x_coords, x_min, side="left"))
    col_stop = int(np.searchsorted(x_coords, x_max, side="right"))

    # Clamp to valid range
    row_start = max(0, row_start)
    row_stop = min(len(y_coords), row_stop)
    col_start = max(0, col_start)
    col_stop = min(len(x_coords), col_stop)

    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError(
            f"Bounding box {bbox} does not overlap the product extent. "
            f"Product x range: [{float(x_coords[0]):.1f}, {float(x_coords[-1]):.1f}], "
            f"y range: [{float(y_coords[0]):.1f}, {float(y_coords[-1]):.1f}] "
            f"(CRS: {crs})"
        )

    sub_x = x_coords[col_start:col_stop]
    sub_y = y_coords[row_start:row_stop]

    logger.info(
        "Bbox → pixel window: rows [%d:%d], cols [%d:%d] (%d × %d pixels)",
        row_start,
        row_stop,
        col_start,
        col_stop,
        row_stop - row_start,
        col_stop - col_start,
    )

    return slice(row_start, row_stop), slice(col_start, col_stop), sub_x, sub_y


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


def _human_size(nbytes: int | float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def estimate_subset_size(
    h5_file: h5py.File,
    product: str,
    frequency: str,
    polarizations: Sequence[str],
    row_slice: slice,
    col_slice: slice,
    layers: Sequence[str] | None = None,
) -> tuple[int, str]:
    """Estimate the download size for a spatial subset.

    Args:
        h5_file: Open HDF5 file handle.
        product: Product type (``"GCOV"``, ``"GSLC"``, ``"GUNW"``, ``"GOFF"``).
        frequency: Frequency label.
        polarizations: Polarization labels to download.
        row_slice: Row slice from :func:`bbox_to_pixel_slices`.
        col_slice: Column slice from :func:`bbox_to_pixel_slices`.
        layers: Layer names (GUNW/GOFF only). If ``None``, uses defaults.

    Returns:
        Tuple of (total_bytes, human_readable_string).
    """
    nrows = row_slice.stop - row_slice.start
    ncols = col_slice.stop - col_slice.start
    total = 0

    for pol in polarizations:
        paths = _get_dataset_paths(product, frequency, pol, layers)
        for ds_path in paths:
            if ds_path in h5_file:
                itemsize = h5_file[ds_path].dtype.itemsize
            else:
                # Estimate based on product type
                itemsize = 8 if product == "GSLC" else 4
            total += nrows * ncols * itemsize

    return total, _human_size(total)


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------


def _get_dataset_paths(
    product: str,
    frequency: str,
    polarization: str,
    layers: Sequence[str] | None = None,
) -> list[str]:
    """Return HDF5 dataset paths for a product/frequency/polarization combination."""
    grid = f"/science/LSAR/{product}/grids/frequency{frequency}"

    if product == "GCOV":
        ds_name = _GCOV_POL_MAP.get(polarization, polarization)
        return [f"{grid}/{ds_name}"]

    if product == "GSLC":
        return [f"{grid}/{polarization}"]

    if product == "GUNW":
        if layers is None:
            layers = ["unwrappedPhase"]
        return [f"{grid}/interferogram/{polarization}/{lay}" for lay in layers]

    if product == "GOFF":
        if layers is None:
            layers = ["alongTrackOffset"]
        return [f"{grid}/pixelOffsets/{polarization}/{lay}" for lay in layers]

    raise ValueError(f"Unsupported product: {product!r}")


def _get_dtype(product: str, layer: str | None = None) -> type:
    """Return the numpy dtype for a product/layer combination."""
    if product == "GSLC":
        return np.complex64
    if product == "GUNW" and layer is not None:
        return _GUNW_LAYER_DTYPES.get(layer, np.float32)
    return np.float32


def _layer_label(ds_path: str) -> str:
    """Extract a short label from a dataset path for filenames."""
    parts = ds_path.rstrip("/").split("/")
    # Return last 1–2 meaningful parts
    return "_".join(parts[-2:]) if len(parts) > 1 else parts[-1]


# ---------------------------------------------------------------------------
# Main subset orchestrator
# ---------------------------------------------------------------------------


def subset_product(
    source: PathType,
    product: str,
    bbox: BBox | str | dict,
    frequency: str = "A",
    polarizations: Sequence[str] | None = None,
    layers: Sequence[str] | None = None,
    output_dir: PathType = Path("nisar_subset"),
    filesystem: fsspec.AbstractFileSystem | None = None,
    confirm: bool = True,
) -> list[Path]:
    """Download a spatial subset of a NISAR product as GeoTIFF(s).

    Opens the remote HDF5 file, reads only the coordinate metadata, computes
    the pixel window for the requested bounding box, reads only those pixels,
    and exports each band as a GeoTIFF.

    Args:
        source: Path or URL to a NISAR HDF5 file.
        product: Product type — ``"GCOV"``, ``"GSLC"``, ``"GUNW"``, or ``"GOFF"``.
        bbox: Bounding box in any format accepted by :func:`parse_bbox`.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarizations: Polarization labels to download. If ``None``, reads
            all available polarizations from the file.
        layers: Layer names for GUNW/GOFF products. If ``None``, defaults to
            ``["unwrappedPhase"]`` for GUNW and ``["alongTrackOffset"]`` for GOFF.
        output_dir: Directory to write GeoTIFF files.
        filesystem: Authenticated filesystem for remote paths.
        confirm: If ``True``, print an estimate and prompt for confirmation
            before downloading. Set ``False`` for non-interactive / GUI use.

    Returns:
        List of paths to the exported GeoTIFF files.

    Raises:
        ValueError: If the product type is unsupported or bbox doesn't overlap.
    """
    product = product.upper()
    if product not in _SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Unsupported product {product!r}. Choose from: {sorted(_SUPPORTED_PRODUCTS)}"
        )

    # Parse bbox from any supported format
    resolved_bbox = parse_bbox(bbox)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_file = open_nisar(source, filesystem=filesystem)
    try:
        # 0. Read identification metadata (date, orbit, track, frame)
        try:
            ident = read_identification(h5_file)
        except (KeyError, ValueError):
            ident = {}
            logger.warning("Could not read identification metadata")

        # Derive a short granule tag for filenames
        source_str = str(source)
        granule_name = Path(source_str.split("?")[0]).stem  # strip query params

        # 1. Read coordinate metadata (tiny transfer)
        crs, full_transform, x_coords, y_coords = get_projection_info_l2(
            h5_file, product, frequency
        )

        # 2. Compute pixel window
        row_slice, col_slice, sub_x, sub_y = bbox_to_pixel_slices(
            x_coords, y_coords, crs, resolved_bbox
        )

        # 3. Resolve polarizations
        if polarizations is None:
            grid_path = f"/science/LSAR/{product}/grids/frequency{frequency}"
            pol_ds = f"{grid_path}/listOfPolarizations"
            polarizations = (  # noqa: SIM108
                list(h5_file[pol_ds][:].astype(str))
                if pol_ds in h5_file
                else ["HH"]
            )
            logger.info("Auto-detected polarizations: %s", polarizations)

        # 4. Estimate size
        total_bytes, size_str = estimate_subset_size(
            h5_file, product, frequency, polarizations, row_slice, col_slice, layers
        )
        nrows = row_slice.stop - row_slice.start
        ncols = col_slice.stop - col_slice.start
        logger.info(
            "Subset: %d × %d pixels, %d band(s), estimated %s",
            nrows,
            ncols,
            len(polarizations),
            size_str,
        )

        # 5. Confirm with user (interactive mode)
        if confirm:
            print(
                f"\nSubset summary:\n"
                f"  Product:       {product}\n"
                f"  Frequency:     {frequency}\n"
                f"  Polarizations: {', '.join(polarizations)}\n"
                f"  Pixel window:  {nrows} × {ncols}\n"
                f"  Est. download: {size_str}\n"
                f"  Output dir:    {output_dir}\n"
            )
            response = input("Proceed? [Y/n] ").strip().lower()
            if response and response not in ("y", "yes"):
                logger.info("Download cancelled by user")
                return []

        # 6. Build sub-transform
        x_spacing = float(sub_x[1] - sub_x[0]) if len(sub_x) > 1 else float(
            full_transform.a
        )
        y_spacing = float(sub_y[1] - sub_y[0]) if len(sub_y) > 1 else float(
            full_transform.e
        )
        sub_transform = Affine(
            x_spacing,
            0.0,
            float(sub_x[0]) - x_spacing / 2.0,
            0.0,
            y_spacing,
            float(sub_y[0]) - y_spacing / 2.0,
        )

        # 7. Read and export each band
        outputs: list[Path] = []

        for pol in polarizations:
            ds_paths = _get_dataset_paths(product, frequency, pol, layers)
            for ds_path in ds_paths:
                if ds_path not in h5_file:
                    logger.warning("Dataset not found, skipping: %s", ds_path)
                    continue

                dataset = h5_file[ds_path]
                logger.info("Reading subset: %s", ds_path)
                raw = dataset[row_slice, col_slice]

                # Determine output dtype
                layer_name = ds_path.split("/")[-1]
                dtype = _get_dtype(product, layer_name)
                data = np.asarray(raw, dtype=dtype)

                # For complex data, export amplitude as float32
                if np.iscomplexobj(data):
                    data = np.abs(data).astype(np.float32)
                    layer_name = f"{layer_name}_amplitude"

                # Build filename — include date if available
                date_tag = ident.get("start_time", "")[:10]  # YYYY-MM-DD
                parts = [product, f"freq{frequency}", pol, layer_name]
                if date_tag:
                    parts.append(date_tag)
                parts.append("subset")
                filename = "_".join(parts) + ".tif"
                out_path = output_dir / filename

                export_geotiff(
                    data,
                    out_path,
                    transform=sub_transform,
                    crs=crs,
                    description=f"{product} {pol} {layer_name}",
                )

                # Write enriched JSON sidecar (overwrites the basic one)
                sidecar = {
                    "source_granule": granule_name,
                    "product_type": product,
                    "frequency": frequency,
                    "polarization": pol,
                    "layer": layer_name,
                    "date": date_tag or None,
                    "start_time": ident.get("start_time"),
                    "end_time": ident.get("end_time"),
                    "orbit": ident.get("orbit"),
                    "track": ident.get("track"),
                    "frame": ident.get("frame"),
                    "crs": str(crs),
                    "epsg": crs.to_epsg(),
                    "bbox_wgs84": {
                        "west": resolved_bbox[0],
                        "south": resolved_bbox[1],
                        "east": resolved_bbox[2],
                        "north": resolved_bbox[3],
                    },
                    "pixel_window": {
                        "row_start": row_slice.start,
                        "row_stop": row_slice.stop,
                        "col_start": col_slice.start,
                        "col_stop": col_slice.stop,
                        "height": nrows,
                        "width": ncols,
                    },
                    "dtype": str(data.dtype),
                    "output": str(out_path),
                }
                sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
                with open(sidecar_path, "w") as fp:
                    json.dump(sidecar, fp, indent=2)

                outputs.append(out_path)
                logger.info("Exported: %s", out_path)

    finally:
        h5_file.close()

    logger.info("Subset complete: %d file(s) written to %s", len(outputs), output_dir)
    return outputs
