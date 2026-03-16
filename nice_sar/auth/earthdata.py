"""Earthdata authentication and S3 filesystem access.

Wraps ``earthaccess`` to provide a consistent authentication interface
and authenticated S3 filesystem for cloud-native data access.
"""

from __future__ import annotations

import logging

import earthaccess
import s3fs

logger = logging.getLogger(__name__)


def login() -> None:
    """Authenticate with NASA Earthdata.

    Uses ``earthaccess.login()`` which checks for credentials in this order:
    1. Environment variables (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``)
    2. ``.netrc`` file
    3. Interactive prompt

    Raises:
        RuntimeError: If authentication fails.
    """
    auth = earthaccess.login()
    if not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata authentication failed. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables, "
            "configure ~/.netrc, or run interactively."
        )
    logger.info("Authenticated with NASA Earthdata")


def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Return an authenticated S3 filesystem for NASA Earthdata Cloud.

    Requires prior authentication via :func:`login`.

    Returns:
        Authenticated ``s3fs.S3FileSystem`` instance.
    """
    credentials = earthaccess.get_s3_credentials(daac="ASF")
    fs = s3fs.S3FileSystem(
        key=credentials["accessKeyId"],
        secret=credentials["secretAccessKey"],
        token=credentials["sessionToken"],
    )
    logger.info("Created authenticated S3 filesystem for ASF DAAC")
    return fs
