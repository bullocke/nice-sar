"""Generate a minimal synthetic NISAR GUNW HDF5 file for offline testing.

Reproduces the HDF5 layout that ``nice_sar.io.products.read_gunw`` relies on.
64×64 pixels at 80 m posting in UTM, matching the synthetic GCOV grid.

Public API
----------
create_synthetic_gunw(path)
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
    grp.create_dataset("productType", data=np.bytes_("GUNW"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-07-01T12:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-07-01T12:00:05"))
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(2345))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_grid(h5: h5py.File, rng: np.random.Generator) -> None:
    freq_grp = h5.create_group("/science/LSAR/GUNW/grids/frequencyA")
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH"], dtype="S2"))
    freq_grp.create_dataset("xCoordinates", data=_X_COORDS.astype(np.float64))
    freq_grp.create_dataset("yCoordinates", data=_Y_COORDS.astype(np.float64))
    freq_grp.create_dataset("xCoordinateSpacing", data=np.float64(_X_SPACING))
    freq_grp.create_dataset("yCoordinateSpacing", data=np.float64(_Y_SPACING))
    proj_ds = freq_grp.create_dataset("projection", data=0)
    proj_ds.attrs["epsg_code"] = _EPSG

    # Interferogram group with HH polarization
    ifg_grp = freq_grp.create_group("interferogram/HH")

    # Unwrapped phase — linear gradient simulating deformation
    phase = np.linspace(0, 6 * np.pi, _NROWS * _NCOLS, dtype=np.float32).reshape(
        _NROWS, _NCOLS
    )
    ifg_grp.create_dataset("unwrappedPhase", data=phase)

    # Coherence magnitude — random in [0.2, 1.0]
    coherence = rng.uniform(0.2, 1.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    ifg_grp.create_dataset("coherenceMagnitude", data=coherence)

    # Connected components — simple block pattern
    cc = np.ones((_NROWS, _NCOLS), dtype=np.uint32)
    cc[:8, :8] = 2  # Small separate component
    ifg_grp.create_dataset("connectedComponents", data=cc)

    # Ionospheric phase screen (not applied by default)
    iono = rng.normal(0, 0.1, size=(_NROWS, _NCOLS)).astype(np.float32)
    ifg_grp.create_dataset("ionospherePhaseScreen", data=iono)

    iono_unc = np.full((_NROWS, _NCOLS), 0.05, dtype=np.float32)
    ifg_grp.create_dataset("ionospherePhaseScreenUncertainty", data=iono_unc)

    # Wrapped interferogram — complex
    wrapped = np.exp(1j * (phase % (2 * np.pi) - np.pi)).astype(np.complex64)
    ifg_grp.create_dataset("wrappedInterferogram", data=wrapped)


def create_synthetic_gunw(path: Path) -> Path:
    """Write a minimal NISAR GUNW HDF5 file to *path*.

    Args:
        path: Destination file path.

    Returns:
        The same *path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(43)
    with h5py.File(path, "w") as h5:
        _write_identification(h5)
        _write_grid(h5, rng)

    return path
