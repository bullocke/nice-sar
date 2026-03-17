"""Product-specific readers for NISAR HDF5 files.

Reads NISAR products into xarray DataArrays with proper coordinates and CRS metadata.
"""

from __future__ import annotations

import logging

import dask.array as da
import fsspec
import h5py
import numpy as np
import xarray as xr
from pyproj import CRS
from rasterio.transform import Affine

from nice_sar._types import PathType
from nice_sar.io.hdf5 import open_nisar

logger = logging.getLogger(__name__)


def read_gcov_metadata(h5_file: h5py.File) -> dict:
    """Read essential metadata from a GCOV product.

    Args:
        h5_file: Open HDF5 file handle.

    Returns:
        Dictionary with product_type, start_time, end_time, orbit, track, frame.
    """
    id_path = "/science/LSAR/identification"
    return {
        "product_type": h5_file[f"{id_path}/productType"][()].decode(),
        "start_time": h5_file[f"{id_path}/zeroDopplerStartTime"][()].decode(),
        "end_time": h5_file[f"{id_path}/zeroDopplerEndTime"][()].decode(),
        "orbit": int(h5_file[f"{id_path}/absoluteOrbitNumber"][()]),
        "track": int(h5_file[f"{id_path}/trackNumber"][()]),
        "frame": int(h5_file[f"{id_path}/frameNumber"][()]),
    }


def get_projection_info(
    h5_file: h5py.File, frequency: str = "A"
) -> tuple[CRS, Affine, np.ndarray, np.ndarray]:
    """Extract projection and georeferencing information from a GCOV product.

    Args:
        h5_file: Open HDF5 file handle.
        frequency: Frequency label (``"A"`` or ``"B"``).

    Returns:
        Tuple of (CRS, Affine transform, x_coordinates, y_coordinates).
    """
    grid_path = f"/science/LSAR/GCOV/grids/frequency{frequency}"
    projection = h5_file[f"{grid_path}/projection"]
    epsg_code = projection.attrs["epsg_code"]
    crs = CRS.from_epsg(int(epsg_code))

    x_coords = h5_file[f"{grid_path}/xCoordinates"][:]
    y_coords = h5_file[f"{grid_path}/yCoordinates"][:]

    x_spacing = h5_file[f"{grid_path}/xCoordinateSpacing"][()]
    y_spacing = h5_file[f"{grid_path}/yCoordinateSpacing"][()]

    x_origin = x_coords[0] - x_spacing / 2.0
    y_origin = y_coords[0] - y_spacing / 2.0

    transform = Affine(x_spacing, 0.0, x_origin, 0.0, y_spacing, y_origin)
    return crs, transform, x_coords, y_coords


def read_gcov(
    source: PathType | h5py.File,
    frequency: str = "A",
    polarization: str = "HH",
    chunks: dict | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> xr.DataArray:
    """Read a GCOV polarization band as a lazy xarray DataArray.

    Args:
        source: Path to NISAR GCOV HDF5 file (local or S3), or an already-open
            ``h5py.File`` handle.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarization: Polarization pair to read. For diagonal terms use e.g.
            ``"HH"`` (reads ``HHHH``); for off-diagonal, use ``"HHHV"`` directly.
        chunks: Dask chunk specification. Defaults to ``{"y": 1024, "x": 1024}``.
        filesystem: Authenticated S3 filesystem. Required when *source* is an
            S3 URI (``s3://...``).

    Returns:
        ``xarray.DataArray`` backed by dask with y/x coordinates and CRS metadata.
    """
    if chunks is None:
        chunks = {"y": 1024, "x": 1024}

    # Accept both path and already-open h5py.File
    owns_file = not isinstance(source, h5py.File)
    h5_file = open_nisar(source, filesystem=filesystem) if owns_file else source

    try:
        crs, transform, x_coords, y_coords = get_projection_info(h5_file, frequency)

        # Map short polarization names to GCOV dataset names
        pol_map = {
            "HH": "HHHH",
            "HV": "HVHV",
            "VV": "VVVV",
            "VH": "VHVH",
        }
        dataset_name = pol_map.get(polarization, polarization)

        grid_path = f"/science/LSAR/GCOV/grids/frequency{frequency}"
        dataset = h5_file[f"{grid_path}/{dataset_name}"]

        # Read the dataset into memory so we don't depend on h5py handle lifetime,
        # then wrap in dask for downstream lazy computation.
        raw = dataset[:].astype(np.float32)
        data = da.from_array(raw, chunks=(chunks["y"], chunks["x"]))

        metadata = read_gcov_metadata(h5_file)
    finally:
        if owns_file:
            h5_file.close()

    da_xr = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "crs": str(crs),
            "transform": tuple(transform),
            "frequency": frequency,
            "polarization": polarization,
            "units": "linear_power",
            **metadata,
        },
    )

    logger.info(
        "Loaded GCOV %s freq%s (%d x %d)",
        polarization,
        frequency,
        len(y_coords),
        len(x_coords),
    )
    return da_xr


def read_quad_covariances(h5_file: h5py.File, grid_path: str) -> dict[str, np.ndarray]:
    """Read quad-pol covariance matrix elements from GCOV.

    Args:
        h5_file: Open HDF5 file handle.
        grid_path: HDF5 group path (e.g., ``/science/LSAR/GCOV/grids/frequencyA``).

    Returns:
        Dictionary with available covariance terms. Keys are ``HHHH``, ``HVHV``,
        ``VVVV`` (real), and ``HHHV``, ``HHVV``, ``HVVV`` (complex).
    """
    out: dict[str, np.ndarray] = {}

    def _get(name: str) -> np.ndarray | None:
        p = f"{grid_path}/{name}"
        if p not in h5_file:
            return None
        arr = h5_file[p][:]
        # h5py ≥3.x may auto-convert compound (r, i) types to numpy complex
        if np.iscomplexobj(arr):
            return arr.astype(np.complex64)
        if (
            hasattr(arr.dtype, "fields")
            and arr.dtype.fields
            and "r" in arr.dtype.fields
            and "i" in arr.dtype.fields
        ):
            return arr["r"].astype(np.float32) + 1j * arr["i"].astype(np.float32)
        return arr.astype(np.float32)

    for k in ("HHHH", "HVHV", "VVVV", "HHHV", "HHVV", "HVVV"):
        val = _get(k)
        if val is not None:
            out[k] = val

    return out
