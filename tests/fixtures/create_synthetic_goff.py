"""Generate a minimal synthetic NISAR GOFF HDF5 file for offline testing.

Reproduces the HDF5 layout that ``nice_sar.io.products.read_goff`` relies on.
64×64 pixels at 80 m posting in UTM.

Public API
----------
create_synthetic_goff(path)
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
_X_SPACING = 80.0
_Y_SPACING = -80.0

_X_COORDS = _X_ORIGIN + np.arange(_NCOLS) * _X_SPACING
_Y_COORDS = _Y_ORIGIN + np.arange(_NROWS) * _Y_SPACING


def _write_identification(h5: h5py.File) -> None:
    grp = h5.create_group("/science/LSAR/identification")
    grp.create_dataset("productType", data=np.bytes_("GOFF"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-08-01T08:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-08-01T08:00:05"))
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(3456))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_grid(h5: h5py.File, rng: np.random.Generator) -> None:
    freq_grp = h5.create_group("/science/LSAR/GOFF/grids/frequencyA")
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH"], dtype="S2"))
    freq_grp.create_dataset("xCoordinates", data=_X_COORDS.astype(np.float64))
    freq_grp.create_dataset("yCoordinates", data=_Y_COORDS.astype(np.float64))
    freq_grp.create_dataset("xCoordinateSpacing", data=np.float64(_X_SPACING))
    freq_grp.create_dataset("yCoordinateSpacing", data=np.float64(_Y_SPACING))
    proj_ds = freq_grp.create_dataset("projection", data=0)
    proj_ds.attrs["epsg_code"] = _EPSG

    # Pixel offsets group with HH polarization
    off_grp = freq_grp.create_group("pixelOffsets/HH")

    # Along-track offset — small random shifts
    along = rng.normal(0, 0.5, size=(_NROWS, _NCOLS)).astype(np.float32)
    off_grp.create_dataset("alongTrackOffset", data=along)

    # Slant-range offset
    slant = rng.normal(0, 0.3, size=(_NROWS, _NCOLS)).astype(np.float32)
    off_grp.create_dataset("slantRangeOffset", data=slant)

    # SNR
    snr = rng.uniform(1.0, 20.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    off_grp.create_dataset("snr", data=snr)


def create_synthetic_goff(path: Path) -> Path:
    """Write a minimal NISAR GOFF HDF5 file to *path*.

    Args:
        path: Destination file path.

    Returns:
        The same *path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(45)
    with h5py.File(path, "w") as h5:
        _write_identification(h5)
        _write_grid(h5, rng)

    return path
