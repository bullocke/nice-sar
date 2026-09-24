"""NASA Earthdata search via earthaccess.

Wraps ``earthaccess.search_data`` for NISAR data discovery by bounding box,
date range, product type, and data maturity.
Public functions:

- :func:`search_earthdata` — Search NASA CMR for NISAR granules using earthaccess
"""

from __future__ import annotations

import logging

import earthaccess

from nice_sar._types import BBox
from nice_sar.search.maturity import nisar_short_names

logger = logging.getLogger(__name__)


def search_earthdata(
    short_name: str | list[str] | None = None,
    bbox: BBox | None = None,
    temporal: tuple[str, str] | None = None,
    count: int = 100,
    product_type: str = "GCOV",
    maturity: str = "provisional",
) -> list:
    """Search NASA Earthdata for NISAR collections.

    By default the collection is resolved from ``product_type`` and
    ``maturity`` (PROVISIONAL GCOV). Passing ``short_name`` overrides both.

    Args:
        short_name: Earthdata collection short name(s). Overrides
            ``product_type`` and ``maturity`` when given.
        bbox: Bounding box as (west, south, east, north).
        temporal: Date range as (start, end) ISO strings.
        count: Maximum number of granules to return.
        product_type: NISAR product type (e.g. ``"GCOV"``, ``"GUNW"``).
        maturity: ``"provisional"`` (default), ``"beta"``, ``"validated"``, or
            ``"any"``.

    Returns:
        List of ``earthaccess`` granule objects.
    """
    if short_name is None:
        names = nisar_short_names(product_type, maturity)
        short_name = names[0] if len(names) == 1 else names

    kwargs: dict = {"short_name": short_name, "count": count}

    if bbox:
        kwargs["bounding_box"] = bbox
    if temporal:
        kwargs["temporal"] = temporal

    logger.info("Searching Earthdata for %s...", short_name)
    results: list = earthaccess.search_data(**kwargs)
    logger.info("Found %d granules", len(results))
    return results
