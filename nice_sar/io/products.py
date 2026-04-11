"""Product-specific readers for NISAR HDF5 files.

Reads NISAR products into xarray DataArrays with proper coordinates and CRS metadata.
Supports GCOV, GSLC, GUNW, GOFF, and RSLC products.
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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def read_identification(h5_file: h5py.File) -> dict:
    """Read product-level identification metadata common to all NISAR products.

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


def get_projection_info_l2(
    h5_file: h5py.File,
    product: str,
    frequency: str = "A",
) -> tuple[CRS, Affine, np.ndarray, np.ndarray]:
    """Extract projection and georeferencing from any geocoded L2 product.

    Works for GCOV, GSLC, GUNW, and GOFF — all share the same coordinate
    structure under ``/science/LSAR/{product}/grids/frequency{A|B}/``.

    Args:
        h5_file: Open HDF5 file handle.
        product: Product type string (e.g., ``"GCOV"``, ``"GUNW"``).
        frequency: Frequency label (``"A"`` or ``"B"``).

    Returns:
        Tuple of (CRS, Affine transform, x_coordinates, y_coordinates).
    """
    grid_path = f"/science/LSAR/{product}/grids/frequency{frequency}"
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


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


def read_gcov_metadata(h5_file: h5py.File) -> dict:
    """Read essential metadata from a GCOV product.

    This is an alias for :func:`read_identification` kept for backward
    compatibility.

    Args:
        h5_file: Open HDF5 file handle.

    Returns:
        Dictionary with product_type, start_time, end_time, orbit, track, frame.
    """
    return read_identification(h5_file)


def get_projection_info(
    h5_file: h5py.File, frequency: str = "A"
) -> tuple[CRS, Affine, np.ndarray, np.ndarray]:
    """Extract projection and georeferencing information from a GCOV product.

    This is an alias for ``get_projection_info_l2(h5, "GCOV", frequency)``
    kept for backward compatibility.

    Args:
        h5_file: Open HDF5 file handle.
        frequency: Frequency label (``"A"`` or ``"B"``).

    Returns:
        Tuple of (CRS, Affine transform, x_coordinates, y_coordinates).
    """
    return get_projection_info_l2(h5_file, "GCOV", frequency)


# ---------------------------------------------------------------------------
# GCOV reader
# ---------------------------------------------------------------------------


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
    if owns_file:  # noqa: SIM108
        h5_file = open_nisar(source, filesystem=filesystem)  # type: ignore[arg-type]
    else:
        h5_file = source  # type: ignore[assignment]

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
        raw = np.asarray(dataset[:], dtype=np.float32)
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
            return np.asarray(arr, dtype=np.complex64)
        if (
            hasattr(arr.dtype, "fields")
            and arr.dtype.fields
            and "r" in arr.dtype.fields
            and "i" in arr.dtype.fields
        ):
            return np.asarray(arr["r"], dtype=np.float32) + 1j * np.asarray(
                arr["i"], dtype=np.float32
            )
        return np.asarray(arr, dtype=np.float32)

    for k in ("HHHH", "HVHV", "VVVV", "HHHV", "HHVV", "HVVV"):
        val = _get(k)
        if val is not None:
            out[k] = val

    return out


# ---------------------------------------------------------------------------
# GSLC reader
# ---------------------------------------------------------------------------

_GSLC_PRODUCT = "GSLC"


def read_gslc(
    source: PathType | h5py.File,
    frequency: str = "A",
    polarization: str = "HH",
    chunks: dict | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> xr.DataArray:
    """Read a GSLC polarization band as a lazy complex xarray DataArray.

    GSLC products contain complex Digital Numbers (DN) on a geocoded grid.
    Phase is flattened for the orbit; amplitude is terrain-corrected but NOT
    radiometrically flattened. ``beta0 = |DN|²``.

    Args:
        source: Path to NISAR GSLC HDF5 file or open ``h5py.File``.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarization: Polarization to read (e.g., ``"HH"``, ``"HV"``).
        chunks: Dask chunk specification. Defaults to ``{"y": 1024, "x": 1024}``.
        filesystem: Authenticated filesystem for remote paths.

    Returns:
        Complex-valued ``xarray.DataArray`` backed by dask.
    """
    if chunks is None:
        chunks = {"y": 1024, "x": 1024}

    owns_file = not isinstance(source, h5py.File)
    if owns_file:
        h5_file = open_nisar(source, filesystem=filesystem)  # type: ignore[arg-type]
    else:
        h5_file = source  # type: ignore[assignment]

    try:
        crs, transform, x_coords, y_coords = get_projection_info_l2(
            h5_file, _GSLC_PRODUCT, frequency
        )
        grid_path = f"/science/LSAR/{_GSLC_PRODUCT}/grids/frequency{frequency}"
        dataset = h5_file[f"{grid_path}/{polarization}"]
        raw = np.asarray(dataset[:], dtype=np.complex64)
        data = da.from_array(raw, chunks=(chunks["y"], chunks["x"]))
        metadata = read_identification(h5_file)
    finally:
        if owns_file:
            h5_file.close()

    da_xr = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        attrs={
            "crs": str(crs),
            "transform": tuple(transform),
            "frequency": frequency,
            "polarization": polarization,
            "units": "complex_dn",
            **metadata,
        },
    )
    logger.info(
        "Loaded GSLC %s freq%s (%d x %d)",
        polarization,
        frequency,
        len(y_coords),
        len(x_coords),
    )
    return da_xr


# ---------------------------------------------------------------------------
# GUNW reader
# ---------------------------------------------------------------------------

_GUNW_PRODUCT = "GUNW"

_GUNW_LAYERS = {
    "unwrappedPhase",
    "coherenceMagnitude",
    "wrappedInterferogram",
    "connectedComponents",
    "ionospherePhaseScreen",
    "ionospherePhaseScreenUncertainty",
}

# Layers stored under /unwrappedInterferogram/ (80 m posting)
_GUNW_80M_LAYERS = {
    "unwrappedPhase",
    "coherenceMagnitude",
    "connectedComponents",
    "ionospherePhaseScreen",
    "ionospherePhaseScreenUncertainty",
}

# Layers stored under /wrappedInterferogram/ (20 m posting)
_GUNW_20M_LAYERS = {
    "wrappedInterferogram",
    "coherenceMagnitude",
}


def _gunw_group_for_layer(layer: str, posting: int) -> str:
    """Return the HDF5 group name for a GUNW layer at a given posting.

    Args:
        layer: Dataset name (e.g. ``"coherenceMagnitude"``).
        posting: Grid posting in metres (``20`` or ``80``).

    Returns:
        Group name: ``"unwrappedInterferogram"`` or ``"wrappedInterferogram"``.

    Raises:
        ValueError: If the layer is not available at the requested posting.
    """
    if posting == 20:
        if layer not in _GUNW_20M_LAYERS:
            raise ValueError(
                f"Layer {layer!r} is not available at 20 m posting. "
                f"Available: {sorted(_GUNW_20M_LAYERS)}"
            )
        return "wrappedInterferogram"
    # 80 m (default)
    if layer not in _GUNW_80M_LAYERS:
        if layer in _GUNW_20M_LAYERS:
            return "wrappedInterferogram"
        raise ValueError(
            f"Layer {layer!r} is not available at 80 m posting. "
            f"Available: {sorted(_GUNW_80M_LAYERS)}"
        )
    return "unwrappedInterferogram"


def _gunw_projection_info(
    h5_file: h5py.File,
    frequency: str,
    group: str,
    polarization: str,
) -> tuple[CRS, Affine, np.ndarray, np.ndarray]:
    """Read projection info from a GUNW per-polarization subgroup.

    Real GUNW files store coordinates per subgroup (each posting has its own
    grid), so we read from the polarization-level group rather than a shared
    top-level coordinate array.
    """
    grp_path = (
        f"/science/LSAR/{_GUNW_PRODUCT}/grids/frequency{frequency}"
        f"/{group}/{polarization}"
    )
    projection = h5_file[f"{grp_path}/projection"]
    epsg_code = projection.attrs["epsg_code"]
    crs = CRS.from_epsg(int(epsg_code))

    x_coords = h5_file[f"{grp_path}/xCoordinates"][:]
    y_coords = h5_file[f"{grp_path}/yCoordinates"][:]
    x_spacing = h5_file[f"{grp_path}/xCoordinateSpacing"][()]
    y_spacing = h5_file[f"{grp_path}/yCoordinateSpacing"][()]

    x_origin = x_coords[0] - x_spacing / 2.0
    y_origin = y_coords[0] - y_spacing / 2.0
    transform = Affine(x_spacing, 0.0, x_origin, 0.0, y_spacing, y_origin)
    return crs, transform, x_coords, y_coords


def read_gunw(
    source: PathType | h5py.File,
    frequency: str = "A",
    polarization: str = "HH",
    layer: str = "unwrappedPhase",
    posting: int | None = None,
    chunks: dict | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> xr.DataArray:
    """Read a GUNW data layer as a lazy xarray DataArray.

    GUNW products contain data at two ground postings:

    * **80 m** (``unwrappedInterferogram/`` group): ``unwrappedPhase``,
      ``coherenceMagnitude``, ``connectedComponents``,
      ``ionospherePhaseScreen``, ``ionospherePhaseScreenUncertainty``.
    * **20 m** (``wrappedInterferogram/`` group): ``wrappedInterferogram``,
      ``coherenceMagnitude``.

    When *layer* exists in only one group the correct posting is inferred
    automatically. For ``coherenceMagnitude`` (present in both), set *posting*
    to ``20`` or ``80`` explicitly; if omitted it defaults to ``80``.

    Only co-polarized (HH or VV) nearest-neighbor 12-day pairs are generated
    by the NISAR mission. The ionospheric phase screen is provided but
    **not applied by default**.

    Args:
        source: Path to NISAR GUNW HDF5 file or open ``h5py.File``.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarization: Co-polarization to read (``"HH"`` or ``"VV"``).
        layer: Dataset to read. One of ``"unwrappedPhase"``,
            ``"coherenceMagnitude"``, ``"wrappedInterferogram"``,
            ``"connectedComponents"``, ``"ionospherePhaseScreen"``,
            ``"ionospherePhaseScreenUncertainty"``.
        posting: Ground posting in metres (``20`` or ``80``). Required only
            when *layer* is ``"coherenceMagnitude"`` (which exists at both
            postings). Defaults to ``80`` when not specified.
        chunks: Dask chunk specification. Defaults to ``{"y": 1024, "x": 1024}``.
        filesystem: Authenticated filesystem for remote paths.

    Returns:
        ``xarray.DataArray`` backed by dask with y/x coordinates and CRS
        metadata. The ``posting`` attribute records the grid posting (20 or 80).

    Raises:
        ValueError: If *layer* is not a recognized GUNW dataset name, or if
            the layer is unavailable at the requested posting.
    """
    if layer not in _GUNW_LAYERS:
        raise ValueError(
            f"Unknown GUNW layer {layer!r}. Choose from: {sorted(_GUNW_LAYERS)}"
        )

    if posting is None:
        posting = 80

    group = _gunw_group_for_layer(layer, posting)

    # Update posting to reflect the actual group when auto-inferred
    # (e.g. wrappedInterferogram only exists at 20 m)
    posting = 20 if group == "wrappedInterferogram" else 80

    if chunks is None:
        chunks = {"y": 1024, "x": 1024}

    owns_file = not isinstance(source, h5py.File)
    if owns_file:
        h5_file = open_nisar(source, filesystem=filesystem)  # type: ignore[arg-type]
    else:
        h5_file = source  # type: ignore[assignment]

    try:
        crs, transform, x_coords, y_coords = _gunw_projection_info(
            h5_file, frequency, group, polarization
        )
        ds_path = (
            f"/science/LSAR/{_GUNW_PRODUCT}/grids/frequency{frequency}"
            f"/{group}/{polarization}/{layer}"
        )
        dataset = h5_file[ds_path]

        if layer == "wrappedInterferogram":
            raw = np.asarray(dataset[:], dtype=np.complex64)
        elif layer == "connectedComponents":
            raw = np.asarray(dataset[:], dtype=np.uint16)
        else:
            raw = np.asarray(dataset[:], dtype=np.float32)

        data = da.from_array(raw, chunks=(chunks["y"], chunks["x"]))
        metadata = read_identification(h5_file)
    finally:
        if owns_file:
            h5_file.close()

    units_map = {
        "unwrappedPhase": "radians",
        "coherenceMagnitude": "unitless",
        "wrappedInterferogram": "complex_phase",
        "connectedComponents": "label",
        "ionospherePhaseScreen": "radians",
        "ionospherePhaseScreenUncertainty": "radians",
    }

    da_xr = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        attrs={
            "crs": str(crs),
            "transform": tuple(transform),
            "frequency": frequency,
            "polarization": polarization,
            "layer": layer,
            "posting": posting,
            "units": units_map.get(layer, "unknown"),
            **metadata,
        },
    )
    logger.info(
        "Loaded GUNW %s/%s freq%s posting=%dm (%d x %d)",
        polarization,
        layer,
        frequency,
        posting,
        len(y_coords),
        len(x_coords),
    )
    return da_xr


# ---------------------------------------------------------------------------
# GOFF reader
# ---------------------------------------------------------------------------

_GOFF_PRODUCT = "GOFF"

_GOFF_LAYERS = {"alongTrackOffset", "slantRangeOffset", "snr"}


def read_goff(
    source: PathType | h5py.File,
    frequency: str = "A",
    polarization: str = "HH",
    layer: str = "alongTrackOffset",
    chunks: dict | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> xr.DataArray:
    """Read a GOFF pixel-offset layer as a lazy xarray DataArray.

    GOFF products contain geocoded pixel offsets at 80 m posting derived from
    cross-correlation of RSLC pairs. Layers may contain outliers — no
    post-processing is applied by the NISAR SDS.

    Args:
        source: Path to NISAR GOFF HDF5 file or open ``h5py.File``.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarization: Polarization to read (e.g., ``"HH"``).
        layer: Dataset to read. One of ``"alongTrackOffset"``,
            ``"slantRangeOffset"``, ``"snr"``.
        chunks: Dask chunk specification. Defaults to ``{"y": 1024, "x": 1024}``.
        filesystem: Authenticated filesystem for remote paths.

    Returns:
        ``xarray.DataArray`` backed by dask.

    Raises:
        ValueError: If *layer* is not a recognized GOFF dataset name.
    """
    if layer not in _GOFF_LAYERS:
        raise ValueError(
            f"Unknown GOFF layer {layer!r}. Choose from: {sorted(_GOFF_LAYERS)}"
        )

    if chunks is None:
        chunks = {"y": 1024, "x": 1024}

    owns_file = not isinstance(source, h5py.File)
    if owns_file:
        h5_file = open_nisar(source, filesystem=filesystem)  # type: ignore[arg-type]
    else:
        h5_file = source  # type: ignore[assignment]

    try:
        crs, transform, x_coords, y_coords = get_projection_info_l2(
            h5_file, _GOFF_PRODUCT, frequency
        )
        off_path = (
            f"/science/LSAR/{_GOFF_PRODUCT}/grids/frequency{frequency}"
            f"/pixelOffsets/{polarization}/{layer}"
        )
        dataset = h5_file[off_path]
        raw = np.asarray(dataset[:], dtype=np.float32)
        data = da.from_array(raw, chunks=(chunks["y"], chunks["x"]))
        metadata = read_identification(h5_file)
    finally:
        if owns_file:
            h5_file.close()

    units_map = {
        "alongTrackOffset": "pixels",
        "slantRangeOffset": "pixels",
        "snr": "ratio",
    }

    da_xr = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        attrs={
            "crs": str(crs),
            "transform": tuple(transform),
            "frequency": frequency,
            "polarization": polarization,
            "layer": layer,
            "units": units_map.get(layer, "unknown"),
            **metadata,
        },
    )
    logger.info(
        "Loaded GOFF %s/%s freq%s (%d x %d)",
        polarization,
        layer,
        frequency,
        len(y_coords),
        len(x_coords),
    )
    return da_xr


# ---------------------------------------------------------------------------
# RSLC reader
# ---------------------------------------------------------------------------

_RSLC_PRODUCT = "RSLC"


def read_rslc(
    source: PathType | h5py.File,
    frequency: str = "A",
    polarization: str = "HH",
    chunks: dict | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> xr.DataArray:
    """Read an RSLC polarization band as a lazy complex xarray DataArray.

    RSLC products are in zero-Doppler range-Doppler geometry (NOT geocoded).
    Complex backscatter values are Digital Numbers with LUTs for radiometric
    calibration. For standard analysis, prefer geocoded products (GCOV/GSLC).

    Args:
        source: Path to NISAR RSLC HDF5 file or open ``h5py.File``.
        frequency: Frequency label (``"A"`` or ``"B"``).
        polarization: Polarization to read (e.g., ``"HH"``, ``"HV"``).
        chunks: Dask chunk specification. Defaults to
            ``{"azimuth": 1024, "range": 1024}``.
        filesystem: Authenticated filesystem for remote paths.

    Returns:
        Complex-valued ``xarray.DataArray`` with azimuth/range dimensions.
    """
    if chunks is None:
        chunks = {"azimuth": 1024, "range": 1024}

    owns_file = not isinstance(source, h5py.File)
    if owns_file:
        h5_file = open_nisar(source, filesystem=filesystem)  # type: ignore[arg-type]
    else:
        h5_file = source  # type: ignore[assignment]

    try:
        swath_path = f"/science/LSAR/{_RSLC_PRODUCT}/swaths/frequency{frequency}"
        dataset = h5_file[f"{swath_path}/{polarization}"]
        raw = np.asarray(dataset[:], dtype=np.complex64)

        slant_range = h5_file[f"{swath_path}/slantRange"][:]
        azimuth_time = h5_file[f"{swath_path}/zeroDopplerTime"][:]

        data = da.from_array(raw, chunks=(chunks["azimuth"], chunks["range"]))
        metadata = read_identification(h5_file)
    finally:
        if owns_file:
            h5_file.close()

    da_xr = xr.DataArray(
        data,
        dims=["azimuth", "range"],
        coords={
            "azimuth": azimuth_time,
            "range": slant_range,
        },
        attrs={
            "frequency": frequency,
            "polarization": polarization,
            "units": "complex_dn",
            **metadata,
        },
    )
    logger.info(
        "Loaded RSLC %s freq%s (%d x %d)",
        polarization,
        frequency,
        len(azimuth_time),
        len(slant_range),
    )
    return da_xr
