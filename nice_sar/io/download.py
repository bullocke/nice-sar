"""Download utilities for NISAR data from ASF.

Provides functions to download NISAR granules from the Alaska Satellite Facility
using ``asf_search`` and ``earthaccess`` for authentication.
Public functions:

- :func:`download_url` — Download a single granule file by direct URL
- :func:`download_granules` — Batch-download a list of ``asf_search`` results
"""

from __future__ import annotations

import logging
from pathlib import Path

import asf_search
import earthaccess

from nice_sar._types import PathType
from nice_sar.search.asf import get_result_size_bytes

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
    skip_existing: bool = True,
) -> list[Path]:
    """Download multiple granules from ASF search results.

    Args:
        results: List of ``asf_search`` result objects.
        output_dir: Local directory to save files.
        session: Authenticated ASF session.
        skip_existing: Skip files that already exist locally with the size
            reported by ASF.

    Returns:
        Paths to the files for ``results``, in the same order (including
        files skipped because they were already present).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    to_fetch: list[str] = []
    for result in results:
        url = result.properties["url"]
        path = output_dir / (result.properties.get("fileName") or url.split("/")[-1])
        paths.append(path)
        if skip_existing and _is_complete(path, get_result_size_bytes(result)):
            logger.info("Already downloaded, skipping: %s", path.name)
            continue
        if path.exists():
            # asf_search will not overwrite an existing (partial) file.
            logger.info("Removing incomplete file: %s", path.name)
            path.unlink()
        to_fetch.append(url)

    if to_fetch:
        if session is None:
            session = _get_asf_session()
        logger.info("Downloading %d granules to %s", len(to_fetch), output_dir)
        asf_search.download_urls(urls=to_fetch, path=str(output_dir), session=session)

    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.warning("%d expected files are missing after download", len(missing))
    logger.info("%d of %d files available", len(paths) - len(missing), len(paths))
    return paths


def _is_complete(path: Path, expected_bytes: int | float | None) -> bool:
    if not path.exists():
        return False
    if expected_bytes is None:
        return True
    return path.stat().st_size == int(expected_bytes)


def _get_asf_session() -> asf_search.ASFSession:
    """Create an authenticated ASF session using earthaccess credentials."""
    auth = earthaccess.login()
    if not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata authentication required. Run nice_sar.auth.login() first."
        )
    token_info = auth.token
    if token_info is None:
        raise RuntimeError("No token available from earthaccess.")
    token = token_info["access_token"]
    session = asf_search.ASFSession()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session
