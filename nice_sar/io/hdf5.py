"""Core HDF5 reader for NISAR products.

Provides transparent access to local and S3-hosted NISAR HDF5 files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import s3fs

from nice_sar._types import PathType

logger = logging.getLogger(__name__)


def open_nisar(
    path: PathType,
    filesystem: s3fs.S3FileSystem | None = None,
) -> h5py.File:
    """Open a NISAR HDF5 file from local disk or S3.

    Args:
        path: Local file path or S3 URI (``s3://bucket/key``).
        filesystem: Authenticated S3 filesystem. Required for S3 paths.

    Returns:
        Open ``h5py.File`` handle in read mode.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If S3 path given without filesystem.
    """
    path_str = str(path)

    if path_str.startswith("s3://"):
        if filesystem is None:
            raise ValueError("An authenticated S3FileSystem is required for S3 paths.")
        logger.info("Opening NISAR file from S3: %s", path_str)
        f = filesystem.open(path_str, "rb")
        return h5py.File(f, "r")

    local_path = Path(path_str)
    if not local_path.exists():
        raise FileNotFoundError(f"NISAR file not found: {local_path}")
    logger.info("Opening NISAR file: %s", local_path)
    return h5py.File(local_path, "r")


def get_frequencies(h5: h5py.File) -> list[str]:
    """Get available frequency bands from a NISAR HDF5 file.

    Args:
        h5: Open NISAR HDF5 file handle.

    Returns:
        List of frequency labels (e.g., ``["A"]`` or ``["A", "B"]``).
    """
    id_path = "/science/LSAR/identification"
    return h5[f"{id_path}/listOfFrequencies"][:].astype(str).tolist()


def get_polarizations(h5: h5py.File, frequency: str = "A") -> list[str]:
    """Get available polarizations for a given frequency.

    Args:
        h5: Open NISAR HDF5 file handle.
        frequency: Frequency label (``"A"`` or ``"B"``).

    Returns:
        List of polarization labels (e.g., ``["HH", "HV"]``).
    """
    for product in ("GCOV", "GSLC", "RSLC"):
        grid_path = f"/science/LSAR/{product}/grids/frequency{frequency}"
        pol_path = f"{grid_path}/listOfPolarizations"
        if pol_path in h5:
            return h5[pol_path][:].astype(str).tolist()
    raise KeyError(f"No polarization list found for frequency {frequency}")
