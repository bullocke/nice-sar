"""ASF DAAC data search via asf_search.

Provides convenience wrappers around ``asf_search`` for discovering
NISAR products by AOI, date range, and product type.
Public functions:

- :func:`search_nisar` — Search NISAR products by AOI, dates, maturity, track, and frame
- :func:`search_gcov` — Convenience wrapper for GCOV-specific searches
- :func:`search_gunw` — Convenience wrapper for GUNW-specific searches
- :func:`get_result_size_bytes` — Extract file size in bytes from an ASF search result
- :func:`summarize_results` — Tabulate maturity, CRID, track, frame, and dates
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import asf_search

from nice_sar._types import BBox
from nice_sar.search.maturity import (
    maturity_from_collection,
    maturity_from_crid,
    nisar_short_names,
    parse_granule_name,
)

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
    maturity: str = "provisional",
    track: int | None = None,
    frame: int | None = None,
    direction: str | None = None,
) -> list:
    """Search for NISAR products on ASF.

    Results are restricted to the collections for the requested data maturity.
    PROVISIONAL (calibrated) data are returned by default; request BETA
    (pre-calibration) data explicitly with ``maturity="beta"``, or both with
    ``maturity="any"``. Avoid mixing maturities in one analysis, since
    differences can arise from changes in the processing software.

    Args:
        product_type: NISAR product type (e.g., ``"GCOV"``, ``"RSLC"``, ``"GUNW"``).
        bbox: Bounding box as (west, south, east, north) in WGS84.
        start: Start date as ISO string or datetime.
        end: End date as ISO string or datetime.
        max_results: Maximum number of results to return.
        maturity: ``"provisional"`` (default), ``"beta"``, ``"validated"``, or
            ``"any"``.
        track: Relative orbit (track) number, 1-173.
        frame: Frame number, 1-176.
        direction: Orbit direction, ``"ASCENDING"``/``"A"`` or
            ``"DESCENDING"``/``"D"``.

    Returns:
        List of ``asf_search`` result objects.
    """
    short_names = nisar_short_names(product_type, maturity)
    search_kwargs: dict[str, Any] = {
        "shortName": short_names,
        "maxResults": max_results,
    }

    if bbox:
        west, south, east, north = bbox
        search_kwargs["intersectsWith"] = (
            f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
        )

    if start:
        search_kwargs["start"] = start
    if end:
        search_kwargs["end"] = end
    if direction:
        search_kwargs["flightDirection"] = _normalize_direction(direction)

    # asf_search's relativeOrbit/frame keywords map to Sentinel-1 attributes,
    # so NISAR track and frame are filtered on the CMR attributes directly.
    cmr_keywords: list[tuple[str, str]] = []
    if track is not None:
        cmr_keywords.append(("attribute[]", f"int,TRACK_NUMBER,{int(track)}"))
    if frame is not None:
        cmr_keywords.append(("attribute[]", f"int,FRAME_NUMBER,{int(frame)}"))
    if cmr_keywords:
        search_kwargs["cmr_keywords"] = cmr_keywords

    logger.info(
        "Searching ASF for NISAR %s products (%s)...",
        product_type,
        ", ".join(short_names),
    )
    results = asf_search.search(**search_kwargs)
    logger.info("Found %d results", len(results))
    return list(results)


def _normalize_direction(direction: str) -> str:
    value = direction.upper()
    if value in ("A", "ASC", "ASCENDING"):
        return "ASCENDING"
    if value in ("D", "DESC", "DESCENDING"):
        return "DESCENDING"
    raise ValueError(f"Unknown orbit direction {direction!r}")


def search_gcov(
    bbox: BBox | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_results: int = 100,
    maturity: str = "provisional",
    **kwargs: Any,
) -> list:
    """Convenience wrapper to search for NISAR GCOV products.

    Args:
        bbox: Bounding box as (west, south, east, north).
        start: Start date.
        end: End date.
        max_results: Maximum results.
        maturity: Data maturity (see :func:`search_nisar`).
        **kwargs: Additional filters passed to :func:`search_nisar`
            (``track``, ``frame``, ``direction``).

    Returns:
        List of search results.
    """
    return search_nisar(
        product_type="GCOV",
        bbox=bbox,
        start=start,
        end=end,
        max_results=max_results,
        maturity=maturity,
        **kwargs,
    )


def search_gunw(
    bbox: BBox | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_results: int = 100,
    maturity: str = "provisional",
    **kwargs: Any,
) -> list:
    """Convenience wrapper to search for NISAR GUNW (interferogram) products.

    Date filters apply to the reference acquisition time.

    Args:
        bbox: Bounding box as (west, south, east, north).
        start: Start date.
        end: End date.
        max_results: Maximum results.
        maturity: Data maturity (see :func:`search_nisar`).
        **kwargs: Additional filters passed to :func:`search_nisar`
            (``track``, ``frame``, ``direction``).

    Returns:
        List of search results.
    """
    return search_nisar(
        product_type="GUNW",
        bbox=bbox,
        start=start,
        end=end,
        max_results=max_results,
        maturity=maturity,
        **kwargs,
    )


@dataclass
class GranuleSummary:
    """Key metadata of a NISAR search result."""

    granule_id: str
    product: str
    maturity: str | None
    crid: str | None
    track: int | None
    direction: str | None
    frame: int | None
    start: str | None
    secondary_start: str | None
    full_frame: bool | None
    size_gb: float | None
    url: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain dictionary."""
        return asdict(self)


def summarize_results(results: list) -> list[GranuleSummary]:
    """Summarize ASF search results for inspection before download.

    Maturity is taken from the collection name when available and otherwise
    inferred from the CRID in the granule name.

    Args:
        results: ``asf_search`` result objects.

    Returns:
        One :class:`GranuleSummary` per result.
    """
    summaries = []
    for result in results:
        props = result.properties
        granule_id = props.get("fileID") or props.get("sceneName") or ""
        try:
            parsed = parse_granule_name(granule_id)
        except ValueError:
            parsed = None

        crid = props.get("crid") or (parsed.crid if parsed else None)
        maturity = maturity_from_collection(props.get("collectionName"))
        if maturity is None and crid:
            maturity = maturity_from_crid(crid)

        size = get_result_size_bytes(result)
        summaries.append(
            GranuleSummary(
                granule_id=granule_id,
                product=props.get("processingLevel") or (parsed.product if parsed else ""),
                maturity=maturity,
                crid=crid,
                track=props.get("pathNumber") or (parsed.track if parsed else None),
                direction=parsed.direction if parsed else None,
                frame=props.get("frameNumber") or (parsed.frame if parsed else None),
                start=parsed.start.isoformat() if parsed else props.get("startTime"),
                secondary_start=(
                    parsed.secondary_start.isoformat()
                    if parsed and parsed.secondary_start
                    else None
                ),
                full_frame=parsed.full_frame if parsed else None,
                size_gb=round(size / 1e9, 2) if size else None,
                url=props.get("url"),
            )
        )
    return summaries
