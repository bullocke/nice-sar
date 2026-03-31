"""Core HDF5 reader for NISAR products.

Provides transparent access to local, S3-hosted, and HTTPS-streamed NISAR HDF5 files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fsspec
import h5py
import s3fs

from nice_sar._types import PathType

logger = logging.getLogger(__name__)


def open_nisar(
    path: PathType,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> h5py.File:
    """Open a NISAR HDF5 file from local disk, S3, or HTTPS.

    Args:
        path: Local file path, S3 URI (``s3://...``), or HTTPS URL.
        filesystem: Authenticated filesystem (``s3fs.S3FileSystem`` or
            ``fsspec`` HTTPS session). Required for S3 and HTTPS paths.

    Returns:
        Open ``h5py.File`` handle in read mode.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If remote path given without filesystem.
    """
    path_str = str(path)

    if path_str.startswith("s3://") or path_str.startswith("https://"):
        if filesystem is None:
            raise ValueError("An authenticated filesystem is required for remote paths.")
        if path_str.startswith("s3://") and not isinstance(filesystem, s3fs.S3FileSystem):
            raise ValueError(
                "Received an s3:// path with a non-S3 filesystem. "
                "Use get_s3_filesystem() for direct S3 reads in AWS us-west-2, "
                "or request an HTTPS granule URL when using get_https_filesystem()."
            )
        if path_str.startswith("https://") and isinstance(filesystem, s3fs.S3FileSystem):
            raise ValueError(
                "Received an https:// path with an S3 filesystem. "
                "Use get_https_filesystem() for HTTPS streaming."
            )
        logger.info("Opening NISAR file from remote: %s", path_str)
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
    result: list[str] = h5[f"{id_path}/listOfFrequencies"][:].astype(str).tolist()
    return result


def get_polarizations(h5: h5py.File, frequency: str = "A") -> list[str]:
    """Get available polarizations for a given frequency.

    Searches across all NISAR product types, including products that use
    ``grids/`` (GCOV, GSLC, GUNW, GOFF) and ``swaths/`` (RSLC).

    For GUNW and GOFF the polarization list is stored at the frequency level,
    same as other geocoded products.

    Args:
        h5: Open NISAR HDF5 file handle.
        frequency: Frequency label (``"A"`` or ``"B"``).

    Returns:
        List of polarization labels (e.g., ``["HH", "HV"]``).
    """
    # Geocoded products under grids/
    for product in ("GCOV", "GSLC", "GUNW", "GOFF"):
        grid_path = f"/science/LSAR/{product}/grids/frequency{frequency}"
        pol_path = f"{grid_path}/listOfPolarizations"
        if pol_path in h5:
            result: list[str] = h5[pol_path][:].astype(str).tolist()
            return result

    # Radar-geometry products under swaths/
    for product in ("RSLC",):
        swath_path = f"/science/LSAR/{product}/swaths/frequency{frequency}"
        pol_path = f"{swath_path}/listOfPolarizations"
        if pol_path in h5:
            result2: list[str] = h5[pol_path][:].astype(str).tolist()
            return result2

    raise KeyError(f"No polarization list found for frequency {frequency}")
