"""Generate a minimal synthetic NISAR GUNW HDF5 file for offline testing.

Reproduces the HDF5 layout of real NISAR GUNW products which store data in
two separate groups at different ground postings:

* ``unwrappedInterferogram/{pol}/`` — 80 m posting (64×64 pixels)
* ``wrappedInterferogram/{pol}/``  — 20 m posting  (256×256 pixels)

Each group carries its own coordinate arrays and projection metadata.

Public API
----------
create_synthetic_gunw(path)
    Write the file to *path*.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# 80 m grid
_NROWS_80 = 64
_NCOLS_80 = 64
_EPSG = 32612

_X_ORIGIN = 425_000.0
_Y_ORIGIN = 4_515_000.0
_X_SPACING_80 = 80.0
_Y_SPACING_80 = -80.0

_X_COORDS_80 = _X_ORIGIN + np.arange(_NCOLS_80) * _X_SPACING_80
_Y_COORDS_80 = _Y_ORIGIN + np.arange(_NROWS_80) * _Y_SPACING_80

# 20 m grid — same spatial extent as the 80 m grid
_NROWS_20 = _NROWS_80 * 4  # 256
_NCOLS_20 = _NCOLS_80 * 4  # 256
_X_SPACING_20 = 20.0
_Y_SPACING_20 = -20.0

_X_COORDS_20 = _X_ORIGIN + np.arange(_NCOLS_20) * _X_SPACING_20
_Y_COORDS_20 = _Y_ORIGIN + np.arange(_NROWS_20) * _Y_SPACING_20


def _write_identification(h5: h5py.File) -> None:
    grp = h5.create_group("/science/LSAR/identification")
    grp.create_dataset("productType", data=np.bytes_("GUNW"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-07-01T12:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-07-01T12:00:05"))
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(2345))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_subgroup_coords(
    grp: h5py.Group,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_spacing: float,
    y_spacing: float,
) -> None:
    """Write per-subgroup coordinate arrays and projection metadata."""
    grp.create_dataset("xCoordinates", data=x_coords.astype(np.float64))
    grp.create_dataset("yCoordinates", data=y_coords.astype(np.float64))
    grp.create_dataset("xCoordinateSpacing", data=np.float64(x_spacing))
    grp.create_dataset("yCoordinateSpacing", data=np.float64(y_spacing))
    proj_ds = grp.create_dataset("projection", data=np.uint32(_EPSG))
    proj_ds.attrs["epsg_code"] = _EPSG


def _write_grid(h5: h5py.File, rng: np.random.Generator) -> None:
    freq_grp = h5.create_group("/science/LSAR/GUNW/grids/frequencyA")
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH"], dtype="S2"))

    # ---- unwrappedInterferogram group (80 m posting) ----
    unwrap_grp = freq_grp.create_group("unwrappedInterferogram/HH")
    _write_subgroup_coords(
        unwrap_grp, _X_COORDS_80, _Y_COORDS_80, _X_SPACING_80, _Y_SPACING_80
    )

    # Unwrapped phase — linear gradient simulating deformation
    phase = np.linspace(
        0, 6 * np.pi, _NROWS_80 * _NCOLS_80, dtype=np.float32
    ).reshape(_NROWS_80, _NCOLS_80)
    unwrap_grp.create_dataset("unwrappedPhase", data=phase)

    # Coherence magnitude (80 m) — random in [0.2, 1.0]
    coherence_80 = rng.uniform(0.2, 1.0, size=(_NROWS_80, _NCOLS_80)).astype(
        np.float32
    )
    unwrap_grp.create_dataset("coherenceMagnitude", data=coherence_80)

    # Connected components — simple block pattern (uint16 per real data)
    cc = np.ones((_NROWS_80, _NCOLS_80), dtype=np.uint16)
    cc[:8, :8] = 2
    unwrap_grp.create_dataset("connectedComponents", data=cc)

    # Ionospheric phase screen (not applied by default)
    iono = rng.normal(0, 0.1, size=(_NROWS_80, _NCOLS_80)).astype(np.float32)
    unwrap_grp.create_dataset("ionospherePhaseScreen", data=iono)

    iono_unc = np.full((_NROWS_80, _NCOLS_80), 0.05, dtype=np.float32)
    unwrap_grp.create_dataset("ionospherePhaseScreenUncertainty", data=iono_unc)

    # ---- wrappedInterferogram group (20 m posting) ----
    wrap_grp = freq_grp.create_group("wrappedInterferogram/HH")
    _write_subgroup_coords(
        wrap_grp, _X_COORDS_20, _Y_COORDS_20, _X_SPACING_20, _Y_SPACING_20
    )

    # Wrapped interferogram — complex
    phase_20 = np.linspace(
        0, 6 * np.pi, _NROWS_20 * _NCOLS_20, dtype=np.float32
    ).reshape(_NROWS_20, _NCOLS_20)
    wrapped = np.exp(1j * (phase_20 % (2 * np.pi) - np.pi)).astype(np.complex64)
    wrap_grp.create_dataset("wrappedInterferogram", data=wrapped)

    # Coherence magnitude (20 m) — random in [0.2, 1.0]
    coherence_20 = rng.uniform(0.2, 1.0, size=(_NROWS_20, _NCOLS_20)).astype(
        np.float32
    )
    wrap_grp.create_dataset("coherenceMagnitude", data=coherence_20)


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
