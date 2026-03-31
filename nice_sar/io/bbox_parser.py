"""Bounding-box parsing utilities.

Accepts bounding boxes in multiple formats (tuple, GeoJSON, file path,
comma-separated string) and normalises to ``BBox`` (west, south, east, north)
in WGS84.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nice_sar._types import BBox

logger = logging.getLogger(__name__)


def validate_bbox(bbox: BBox) -> BBox:
    """Validate a bounding box and return it unchanged.

    Args:
        bbox: Bounding box as (west, south, east, north) in WGS84 degrees.

    Returns:
        The validated bounding box.

    Raises:
        ValueError: If the bounding box is geometrically invalid.
    """
    west, south, east, north = bbox

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError(
            f"Longitude values must be in [-180, 180], got west={west}, east={east}"
        )
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError(
            f"Latitude values must be in [-90, 90], got south={south}, north={north}"
        )
    if south >= north:
        raise ValueError(f"South ({south}) must be less than north ({north})")
    if west >= east:
        raise ValueError(
            f"West ({west}) must be less than east ({east}). "
            "Anti-meridian-crossing boxes are not supported."
        )
    return bbox


def parse_bbox(source: Any) -> BBox:
    """Parse a bounding box from various input formats.

    Supported formats:
    - ``BBox`` tuple ``(west, south, east, north)``
    - GeoJSON geometry dict (Polygon or MultiPolygon — extracts envelope)
    - GeoJSON geometry as a JSON string
    - Path (str or Path) to a spatial file (``.geojson``, ``.shp``, ``.gpkg``)
    - Comma-separated string ``"west,south,east,north"``

    All outputs are in **WGS84** (EPSG:4326).

    Args:
        source: Bounding box in any of the supported formats.

    Returns:
        Validated ``BBox`` tuple ``(west, south, east, north)``.

    Raises:
        ValueError: If the source cannot be parsed or the resulting bbox is invalid.
    """
    # 1. Already a tuple/list of 4 floats
    if isinstance(source, (tuple, list)) and len(source) == 4:
        try:
            bbox = tuple(float(v) for v in source)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot convert bbox values to float: {source}") from exc
        return validate_bbox(bbox)  # type: ignore[arg-type]

    # 2. Dict — treat as GeoJSON geometry
    if isinstance(source, dict):
        return _bbox_from_geojson_geometry(source)

    # 3. String — could be JSON, CSV, or a file path
    if isinstance(source, (str, Path)):
        source_str = str(source)

        # Try JSON first
        if source_str.lstrip().startswith("{"):
            try:
                geojson = json.loads(source_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"String looks like JSON but cannot be parsed: {exc}") from exc
            return _bbox_from_geojson_geometry(geojson)

        # Try as file path
        path = Path(source_str)
        if path.suffix.lower() in (".geojson", ".json", ".shp", ".gpkg"):
            return _bbox_from_file(path)

        # Try comma-separated "w,s,e,n"
        parts = source_str.split(",")
        if len(parts) == 4:
            try:
                bbox = tuple(float(p.strip()) for p in parts)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse comma-separated bbox from {source_str!r}: {exc}"
                ) from exc
            return validate_bbox(bbox)  # type: ignore[arg-type]

        raise ValueError(
            f"Cannot parse bbox from string: {source_str!r}. "
            "Expected GeoJSON, a file path, or 'west,south,east,north'."
        )

    raise TypeError(f"Unsupported bbox source type: {type(source).__name__}")


def _bbox_from_geojson_geometry(geojson: dict) -> BBox:
    """Extract the bounding-box envelope from a GeoJSON geometry or Feature."""
    # Handle Feature wrapper
    if geojson.get("type") == "Feature":
        geom = geojson.get("geometry")
        if geom is None:
            raise ValueError("GeoJSON Feature has no geometry")
        return _bbox_from_geojson_geometry(geom)

    # Handle FeatureCollection
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection has no features")
        # Union of all feature bboxes
        bboxes = [_bbox_from_geojson_geometry(f) for f in features]
        west = min(b[0] for b in bboxes)
        south = min(b[1] for b in bboxes)
        east = max(b[2] for b in bboxes)
        north = max(b[3] for b in bboxes)
        return validate_bbox((west, south, east, north))

    coords = geojson.get("coordinates")
    if coords is None:
        raise ValueError("GeoJSON geometry has no 'coordinates' key")

    # Flatten all coordinate pairs
    flat = _flatten_coords(coords)
    if not flat:
        raise ValueError("No coordinates found in GeoJSON geometry")

    lons = [c[0] for c in flat]
    lats = [c[1] for c in flat]
    return validate_bbox((min(lons), min(lats), max(lons), max(lats)))


def _flatten_coords(obj: Any) -> list[list[float]]:
    """Recursively flatten nested coordinate arrays to a list of [lon, lat] pairs."""
    if not isinstance(obj, (list, tuple)):
        return []
    # If the first element is a number, this is a coordinate pair
    if obj and isinstance(obj[0], (int, float)):
        return [list(obj[:2])]
    result: list[list[float]] = []
    for item in obj:
        result.extend(_flatten_coords(item))
    return result


def _bbox_from_file(path: Path) -> BBox:
    """Read a spatial file and return its WGS84 bounding box."""
    if not path.exists():
        raise FileNotFoundError(f"Bbox file not found: {path}")

    # For .geojson/.json files, try pure-JSON parsing first (no geopandas needed)
    if path.suffix.lower() in (".geojson", ".json"):
        text = path.read_text(encoding="utf-8")
        geojson = json.loads(text)
        # If it's a simple geometry or Feature, use the fast path
        if geojson.get("type") in ("Polygon", "MultiPolygon", "Point", "Feature"):
            return _bbox_from_geojson_geometry(geojson)
        if geojson.get("type") == "FeatureCollection":
            return _bbox_from_geojson_geometry(geojson)

    # Fall back to geopandas for .shp, .gpkg, or complex files
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            f"geopandas is required to read {path.suffix} files. "
            "Install it with: pip install geopandas"
        ) from exc

    gdf = gpd.read_file(path)
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    return validate_bbox((float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])))
