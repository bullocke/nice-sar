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


# NISAR-specific S3 credential endpoint (using daac='ASF' will fail for NISAR;
# see https://github.com/earthaccess-dev/earthaccess/issues/1184).
_NISAR_S3_ENDPOINT = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"


def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Return an authenticated S3 filesystem for NISAR data in Earthdata Cloud.

    Uses the NISAR-specific S3 credential endpoint. Direct S3 access is only
    available from within AWS ``us-west-2`` (e.g., CryoCloud, EC2). For access
    outside that region, use :func:`get_https_filesystem` instead.

    Requires prior authentication via :func:`login`.

    Returns:
        Authenticated ``s3fs.S3FileSystem`` instance.
    """
    auth = earthaccess.login()
    credentials = auth.get_s3_credentials(endpoint=_NISAR_S3_ENDPOINT)
    fs = s3fs.S3FileSystem(
        key=credentials["accessKeyId"],
        secret=credentials["secretAccessKey"],
        token=credentials["sessionToken"],
    )
    logger.info("Created authenticated S3 filesystem for NISAR")
    return fs


def get_https_filesystem() -> "fsspec.AbstractFileSystem":
    """Return an authenticated HTTPS filesystem for NISAR data.

    Works from anywhere (Colab, local, CHPC) — no AWS region restriction.
    Requires prior authentication via :func:`login`.

    Returns:
        Authenticated ``fsspec`` HTTPS filesystem.
    """
    fs = earthaccess.get_fsspec_https_session()
    logger.info("Created authenticated HTTPS filesystem for NISAR")
    return fs
