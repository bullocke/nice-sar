"""Generate a minimal synthetic NISAR RSLC HDF5 file for offline testing.

Reproduces the HDF5 layout that ``nice_sar.io.products.read_rslc`` relies on.
64×64 pixels on a range-Doppler grid (NOT geocoded).

Public API
----------
create_synthetic_rslc(path)
    Write the file to *path*.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

_NROWS = 64  # azimuth lines
_NCOLS = 64  # range samples

# Simulated range-Doppler grid parameters
_SLANT_RANGE_START = 800_000.0  # metres
_SLANT_RANGE_SPACING = 3.0  # metres (range pixel spacing)
_AZIMUTH_TIME_START = 0.0  # seconds from epoch
_AZIMUTH_TIME_SPACING = 0.002  # seconds (~500 Hz PRF)


def _write_identification(h5: h5py.File) -> None:
    grp = h5.create_group("/science/LSAR/identification")
    grp.create_dataset("productType", data=np.bytes_("RSLC"))
    grp.create_dataset("zeroDopplerStartTime", data=np.bytes_("2025-06-01T12:00:00"))
    grp.create_dataset("zeroDopplerEndTime", data=np.bytes_("2025-06-01T12:00:05"))
    grp.create_dataset("absoluteOrbitNumber", data=np.int32(1234))
    grp.create_dataset("trackNumber", data=np.int32(42))
    grp.create_dataset("frameNumber", data=np.int32(100))
    grp.create_dataset("listOfFrequencies", data=np.array(["A"], dtype="S1"))


def _write_swath(h5: h5py.File, rng: np.random.Generator) -> None:
    freq_grp = h5.create_group("/science/LSAR/RSLC/swaths/frequencyA")
    freq_grp.create_dataset("listOfPolarizations", data=np.array(["HH", "HV"], dtype="S2"))

    # Range-Doppler coordinates
    slant_range = _SLANT_RANGE_START + np.arange(_NCOLS) * _SLANT_RANGE_SPACING
    azimuth_time = _AZIMUTH_TIME_START + np.arange(_NROWS) * _AZIMUTH_TIME_SPACING
    freq_grp.create_dataset("slantRange", data=slant_range.astype(np.float64))
    freq_grp.create_dataset("zeroDopplerTime", data=azimuth_time.astype(np.float64))

    # Complex SLC data — random complex values
    amplitude = rng.exponential(0.1, size=(_NROWS, _NCOLS)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(_NROWS, _NCOLS)).astype(np.float32)
    hh = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    freq_grp.create_dataset("HH", data=hh)

    amplitude_hv = rng.exponential(0.03, size=(_NROWS, _NCOLS)).astype(np.float32)
    phase_hv = rng.uniform(-np.pi, np.pi, size=(_NROWS, _NCOLS)).astype(np.float32)
    hv = (amplitude_hv * np.exp(1j * phase_hv)).astype(np.complex64)
    freq_grp.create_dataset("HV", data=hv)


def create_synthetic_rslc(path: Path) -> Path:
    """Write a minimal NISAR RSLC HDF5 file to *path*.

    Args:
        path: Destination file path.

    Returns:
        The same *path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(46)
    with h5py.File(path, "w") as h5:
        _write_identification(h5)
        _write_swath(h5, rng)

    return path
