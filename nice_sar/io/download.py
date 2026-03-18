"""Download utilities for NISAR data from ASF.

Provides functions to download NISAR granules from the Alaska Satellite Facility
using ``asf_search`` and ``earthaccess`` for authentication.
"""

from __future__ import annotations

import logging
from pathlib import Path

import asf_search
import earthaccess

from nice_sar._types import PathType

logger = logging.getLogger(__name__)


def download_url(
    url: str,
    output_dir: PathType,
    session: asf_search.ASFSession | None = None,
) -> Path:
    """Download a single file from ASF by URL.

    Args:
        url: Direct URL to the NISAR product file.
        output_dir: Local directory to save the file.
        session: Authenticated ASF session. If ``None``, creates one via
            earthaccess credentials.

    Returns:
        Path to the downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_asf_session()

    logger.info("Downloading: %s", url.split("/")[-1])
    asf_search.download_url(url=url, path=str(output_dir), session=session)

    filename = url.split("/")[-1]
    result = output_dir / filename
    logger.info("Downloaded: %s", result)
    return result


def download_granules(
    results: list,
    output_dir: PathType,
    session: asf_search.ASFSession | None = None,
) -> list[Path]:
    """Download multiple granules from ASF search results.

    Args:
        results: List of ``asf_search`` result objects.
        output_dir: Local directory to save files.
        session: Authenticated ASF session.

    Returns:
        List of paths to downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_asf_session()

    logger.info("Downloading %d granules to %s", len(results), output_dir)
    asf_search.download_urls(
        urls=[r.properties["url"] for r in results],
        path=str(output_dir),
        session=session,
    )

    downloaded = list(output_dir.glob("*.h5"))
    logger.info("Downloaded %d files", len(downloaded))
    return downloaded


def _get_asf_session() -> asf_search.ASFSession:
    """Create an authenticated ASF session using earthaccess credentials."""
    auth = earthaccess.login()
    if not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata authentication required. "
            "Run nice_sar.auth.login() first."
        )
    token_info = auth.token
    if token_info is None:
        raise RuntimeError("No token available from earthaccess.")
    token = token_info["access_token"]
    session = asf_search.ASFSession()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session
