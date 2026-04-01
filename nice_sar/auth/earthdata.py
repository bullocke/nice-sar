"""Earthdata authentication and S3 filesystem access.

Wraps ``earthaccess`` to provide a consistent authentication interface
and authenticated S3 filesystem for cloud-native data access.
Public functions:

- :func:`login` — Authenticate with NASA Earthdata (interactive or .netrc)
- :func:`get_s3_filesystem` — Return an authenticated ``s3fs.S3FileSystem`` for direct S3 access
- :func:`get_https_filesystem` — Return an authenticated fsspec filesystem for HTTPS access
- :func:`get_granule_url` — Resolve the preferred access URL for a granule result
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import earthaccess
import fsspec
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


def _can_use_s3_direct_access() -> bool:
    """Return whether the runtime can use direct S3 access for NISAR."""
    return os.environ.get("AWS_DEFAULT_REGION") == "us-west-2"


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


def get_https_filesystem() -> fsspec.AbstractFileSystem:
    """Return an authenticated HTTPS filesystem for NISAR data.

    Works from anywhere (Colab, local, CHPC) — no AWS region restriction.
    Requires prior authentication via :func:`login`.

    Returns:
        Authenticated ``fsspec`` HTTPS filesystem.
    """
    fs = earthaccess.get_fsspec_https_session()
    logger.info("Created authenticated HTTPS filesystem for NISAR")
    return fs


def get_granule_url(
    granule: Any,
    access: Literal["auto", "s3", "https"] = "auto",
) -> str:
    """Return a NISAR granule URL compatible with the selected transport.

    Args:
        granule: ``earthaccess`` granule object returned by ``search_data``.
        access: Transport to prefer. ``"auto"`` selects ``"s3"`` only when
            running in AWS ``us-west-2``; otherwise it selects ``"https"``.

    Returns:
        Compatible ``s3://`` or ``https://`` URL for the granule.

    Raises:
        ValueError: If no compatible URL is available for the requested access mode.
    """
    resolved_access = access
    if resolved_access == "auto":
        resolved_access = "s3" if _can_use_s3_direct_access() else "https"

    desired_scheme = "s3://" if resolved_access == "s3" else "https://"
    link_sets = {
        "direct": earthaccess.results.DataGranule.data_links(granule, access="direct") or [],
        "external": earthaccess.results.DataGranule.data_links(granule, access="external") or [],
    }

    access_order = ("direct", "external") if resolved_access == "s3" else ("external", "direct")
    for access_kind in access_order:
        for link in link_sets[access_kind]:
            if isinstance(link, str) and link.startswith(desired_scheme):
                logger.info("Selected %s granule URL for %s access", desired_scheme, resolved_access)
                return link

    available_schemes = sorted(
        {
            link.split("://", 1)[0]
            for links in link_sets.values()
            for link in links
            if isinstance(link, str) and "://" in link
        }
    )
    raise ValueError(
        "No compatible granule URL found for "
        f"{resolved_access!r} access. Available schemes: {available_schemes or ['none']}."
    )
