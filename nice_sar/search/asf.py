"""ASF DAAC data search via asf_search.

Provides convenience wrappers around ``asf_search`` for discovering
NISAR products by AOI, date range, and product type.
Public functions:

- :func:`search_nisar` — Search for any NISAR product type by AOI, date range, and beam mode
- :func:`search_gcov` — Convenience wrapper for GCOV-specific searches
- :func:`get_result_size_bytes` — Extract file size in bytes from an ASF search result
"""

from __future__ import annotations

import logging
from datetime import datetime

import asf_search

from nice_sar._types import BBox

logger = logging.getLogger(__name__)


def get_result_size_bytes(result: object) -> int | float | None:
    """Extract a granule size in bytes from an ASF search result.

    ASF metadata is not uniform across products. Most products expose a scalar
    ``bytes`` property, while current NISAR products expose ``bytes`` as a
    mapping keyed by filename with per-file metadata.

    Args:
        result: ASF search result object or a properties mapping.

    Returns:
        Size in bytes for the primary data file, if available.
    """
    props = result if isinstance(result, dict) else getattr(result, "properties", None)
    if not isinstance(props, dict):
        return None

    size_info = props.get("bytes")
    if isinstance(size_info, int | float):
        return size_info

    if not isinstance(size_info, dict):
        return None

    file_name = props.get("fileName")
    if isinstance(file_name, str):
        file_entry = size_info.get(file_name)
        if isinstance(file_entry, dict):
            nested_bytes = file_entry.get("bytes")
            if isinstance(nested_bytes, int | float):
                return nested_bytes
        elif isinstance(file_entry, int | float):
            return file_entry

    nested_bytes = size_info.get("bytes")
    if isinstance(nested_bytes, int | float):
        return nested_bytes

    if len(size_info) == 1:
        only_entry = next(iter(size_info.values()))
        if isinstance(only_entry, dict):
            nested_bytes = only_entry.get("bytes")
            if isinstance(nested_bytes, int | float):
                return nested_bytes
        elif isinstance(only_entry, int | float):
            return only_entry

    return None


def search_nisar(
    product_type: str = "GCOV",
    bbox: BBox | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_results: int = 100,
) -> list:
    """Search for NISAR products on ASF.

    Args:
        product_type: NISAR product type (e.g., ``"GCOV"``, ``"RSLC"``, ``"GUNW"``).
        bbox: Bounding box as (west, south, east, north) in WGS84.
        start: Start date as ISO string or datetime.
        end: End date as ISO string or datetime.
        max_results: Maximum number of results to return.

    Returns:
        List of ``asf_search`` result objects.
    """
    search_kwargs: dict = {
        "dataset": "NISAR",
        "maxResults": max_results,
    }

    if product_type:
        search_kwargs["processingLevel"] = product_type

    if bbox:
        west, south, east, north = bbox
        search_kwargs["intersectsWith"] = (
            f"POLYGON(({west} {south},{east} {south},"
            f"{east} {north},{west} {north},{west} {south}))"
        )

    if start:
        search_kwargs["start"] = start
    if end:
        search_kwargs["end"] = end

    logger.info("Searching ASF for NISAR %s products...", product_type)
    results = asf_search.search(**search_kwargs)
    logger.info("Found %d results", len(results))
    return list(results)


def search_gcov(
    bbox: BBox | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_results: int = 100,
) -> list:
    """Convenience wrapper to search for NISAR GCOV products.

    Args:
        bbox: Bounding box as (west, south, east, north).
        start: Start date.
        end: End date.
        max_results: Maximum results.

    Returns:
        List of search results.
    """
    return search_nisar(
        product_type="GCOV",
        bbox=bbox,
        start=start,
        end=end,
        max_results=max_results,
    )
