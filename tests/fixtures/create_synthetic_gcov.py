"""Generate a minimal synthetic NISAR GCOV HDF5 file for offline testing.

The file reproduces the HDF5 group/dataset/attribute layout that
`nice_sar.io.hdf5` and `nice_sar.io.products` rely on, at 64×64 pixels.

Public API
----------
create_synthetic_gcov(path)
    Write the file to *path*.  Safe to call multiple times (recreates).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# Constants matching the NISAR GCOV schema used by nice_sar.io
# --------------------------------------------------------------------------- #
_NROWS = 64
_NCOLS = 64
_EPSG = 32612  # UTM zone 12N — Salt Lake City area

# Fake UTM coordinates (metre grid, 30 m spacing)
_X_ORIGIN = 425_000.0
_Y_ORIGIN = 4_515_000.0
_X_SPACING = 30.0
_Y_SPACING = -30.0  # negative = north-up

_X_COORDS = _X_ORIGIN + np.arange(_NCOLS) * _X_SPACING
_Y_COORDS = _Y_ORIGIN + np.arange(_NROWS) * _Y_SPACING


def _write_identification(h5: h5py.File) -> None:
    """Populate ``/science/LSAR/identification``."""
    grp = h5.create_group("/science/LSAR/identification")

    # Scalar byte-string datasets (decoded in read_gcov_metadata)
    grp.create_dataset("productType", data=np.bytes_("GCOV"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-06-01T12:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-06-01T12:00:05"))

    # Scalar integers
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(1234))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))

    # Frequency list — fixed-length byte strings
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_grid(h5: h5py.File, rng: np.random.Generator) -> None:
    """Populate ``/science/LSAR/GCOV/grids/frequencyA``."""
    freq_grp = h5.create_group("/science/LSAR/GCOV/grids/frequencyA")

    # Polarization list
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH", "HV"], dtype="S2"))

    # Coordinate arrays
    freq_grp.create_dataset("xCoordinates", data=_X_COORDS.astype(np.float64))
    freq_grp.create_dataset("yCoordinates", data=_Y_COORDS.astype(np.float64))

    # Coordinate spacing (scalar)
    freq_grp.create_dataset("xCoordinateSpacing", data=np.float64(_X_SPACING))
    freq_grp.create_dataset("yCoordinateSpacing", data=np.float64(_Y_SPACING))

    # Projection dataset with epsg_code attribute (used by get_projection_info)
    proj_ds = freq_grp.create_dataset("projection", data=0)
    proj_ds.attrs["epsg_code"] = _EPSG

    # Diagonal covariance terms — real float32, exponential distribution
    hhhh = rng.exponential(0.05, size=(_NROWS, _NCOLS)).astype(np.float32)
    hvhv = rng.exponential(0.01, size=(_NROWS, _NCOLS)).astype(np.float32)
    freq_grp.create_dataset("HHHH", data=hhhh)
    freq_grp.create_dataset("HVHV", data=hvhv)

    # Off-diagonal covariance terms — compound type with 'r' and 'i' fields
    complex_dt = np.dtype([("r", np.float32), ("i", np.float32)])
    hhhv = np.zeros((_NROWS, _NCOLS), dtype=complex_dt)
    hhhv["r"] = rng.normal(0, 0.005, size=(_NROWS, _NCOLS)).astype(np.float32)
    hhhv["i"] = rng.normal(0, 0.005, size=(_NROWS, _NCOLS)).astype(np.float32)
    freq_grp.create_dataset("HHHV", data=hhhv)


def create_synthetic_gcov(path: Path) -> Path:
    """Write a minimal NISAR GCOV HDF5 file to *path*.

    Args:
        path: Destination file path. Parent directories are created automatically.

    Returns:
        The same *path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as h5:
        _write_identification(h5)
        _write_grid(h5, rng)

    return path
