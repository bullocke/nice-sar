"""Generate a minimal synthetic NISAR GSLC HDF5 file for offline testing.

Reproduces the HDF5 layout that ``nice_sar.io.products.read_gslc`` relies on.
64×64 pixels at 5 m posting in UTM.

Public API
----------
create_synthetic_gslc(path)
    Write the file to *path*.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

_NROWS = 64
_NCOLS = 64
_EPSG = 32612

_X_ORIGIN = 425_000.0
_Y_ORIGIN = 4_515_000.0
_X_SPACING = 5.0
_Y_SPACING = -5.0

_X_COORDS = _X_ORIGIN + np.arange(_NCOLS) * _X_SPACING
_Y_COORDS = _Y_ORIGIN + np.arange(_NROWS) * _Y_SPACING


def _write_identification(h5: h5py.File) -> None:
    grp = h5.create_group("/science/LSAR/identification")
    grp.create_dataset("productType", data=np.bytes_("GSLC"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-06-15T10:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-06-15T10:00:05"))
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(1500))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_grid(h5: h5py.File, rng: np.random.Generator) -> None:
    freq_grp = h5.create_group("/science/LSAR/GSLC/grids/frequencyA")
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH", "HV"], dtype="S2"))
    freq_grp.create_dataset("xCoordinates", data=_X_COORDS.astype(np.float64))
    freq_grp.create_dataset("yCoordinates", data=_Y_COORDS.astype(np.float64))
    freq_grp.create_dataset("xCoordinateSpacing", data=np.float64(_X_SPACING))
    freq_grp.create_dataset("yCoordinateSpacing", data=np.float64(_Y_SPACING))
    proj_ds = freq_grp.create_dataset("projection", data=0)
    proj_ds.attrs["epsg_code"] = _EPSG

    # Complex DN values — amplitude ~ exponential, phase ~ uniform
    amplitude = rng.exponential(0.1, size=(_NROWS, _NCOLS)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(_NROWS, _NCOLS)).astype(np.float32)
    hh = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    freq_grp.create_dataset("HH", data=hh)

    amplitude_hv = rng.exponential(0.03, size=(_NROWS, _NCOLS)).astype(np.float32)
    phase_hv = rng.uniform(-np.pi, np.pi, size=(_NROWS, _NCOLS)).astype(np.float32)
    hv = (amplitude_hv * np.exp(1j * phase_hv)).astype(np.complex64)
    freq_grp.create_dataset("HV", data=hv)


def create_synthetic_gslc(path: Path) -> Path:
    """Write a minimal NISAR GSLC HDF5 file to *path*.

    Args:
        path: Destination file path.

    Returns:
        The same *path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(44)
    with h5py.File(path, "w") as h5:
        _write_identification(h5)
        _write_grid(h5, rng)

    return path
