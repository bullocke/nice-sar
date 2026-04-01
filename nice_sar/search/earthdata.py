"""NASA Earthdata search via earthaccess.

Wraps ``earthaccess.search_data`` for NISAR data discovery by bounding box,
date range, and short name.
Public functions:

- :func:`search_earthdata` — Search NASA CMR for NISAR granules using earthaccess
"""

from __future__ import annotations

import logging

import earthaccess

from nice_sar._types import BBox

logger = logging.getLogger(__name__)


def search_earthdata(
    short_name: str = "NISAR_L2_GCOV_BETA_V1",
    bbox: BBox | None = None,
    temporal: tuple[str, str] | None = None,
    count: int = 100,
) -> list:
    """Search NASA Earthdata for NISAR collections.

    Args:
        short_name: Earthdata collection short name.
        bbox: Bounding box as (west, south, east, north).
        temporal: Date range as (start, end) ISO strings.
        count: Maximum number of granules to return.

    Returns:
        List of ``earthaccess`` granule objects.
    """
    kwargs: dict = {"short_name": short_name, "count": count}

    if bbox:
        kwargs["bounding_box"] = bbox
    if temporal:
        kwargs["temporal"] = temporal

    logger.info("Searching Earthdata for %s...", short_name)
    results: list = earthaccess.search_data(**kwargs)
    logger.info("Found %d granules", len(results))
    return results
