"""Interactive map visualization with folium.

Provides utilities for overlaying SAR rasters on interactive web maps,
including coherence, displacement, and RGB composites.  Requires the
optional ``folium`` dependency (``pip install nice-sar[viz]``).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


def _check_folium() -> Any:
    """Lazily import folium or raise a helpful error."""
    try:
        import folium
    except ImportError as exc:
        raise ImportError(
            "folium is required for interactive mapping. "
            "Install it with: pip install nice-sar[viz]"
        ) from exc
    return folium


def _to_latlon_bounds(
    crs: CRS | str | int,
    transform: Affine | tuple,
    height: int,
    width: int,
) -> list[list[float]]:
    """Convert raster extent to lat/lon bounds for folium.

    Returns ``[[south, west], [north, east]]`` in EPSG:4326.
    """
    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)
    if not isinstance(transform, Affine):
        transform = Affine(*transform)

    # Raster corner coordinates in source CRS
    x_min = transform.c
    y_max = transform.f
    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e  # e is negative for north-up

    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_min, lat_min = transformer.transform(x_min, y_min)
    lon_max, lat_max = transformer.transform(x_max, y_max)

    return [[lat_min, lon_min], [lat_max, lon_max]]


def _array_to_png(data: np.ndarray) -> bytes:
    """Encode a uint8 RGBA array (H, W, 4) to PNG bytes."""
    from PIL import Image

    img = Image.fromarray(data, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_colormap(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    nodata_alpha: bool = True,
) -> np.ndarray:
    """Map float data to RGBA uint8 via a matplotlib colormap.

    Args:
        data: 2D float array.
        vmin: Min value for normalization.
        vmax: Max value for normalization.
        cmap: Matplotlib colormap name.
        nodata_alpha: If True, set NaN pixels to transparent.

    Returns:
        ``(H, W, 4)`` uint8 RGBA array.
    """
    import matplotlib.pyplot as plt

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.colormaps.get_cmap(cmap)
    rgba = (mapper(norm(data)) * 255).astype(np.uint8)

    if nodata_alpha:
        rgba[np.isnan(data), 3] = 0

    return rgba


def overlay_raster(
    data: np.ndarray,
    crs: CRS | str | int,
    transform: Affine | tuple,
    *,
    map_obj: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    name: str = "SAR layer",
    opacity: float = 0.7,
    zoom_start: int = 10,
) -> Any:
    """Overlay a 2-D raster on a folium map.

    Args:
        data: 2-D float array (e.g. coherence, displacement, backscatter).
        crs: Coordinate reference system of the raster.
        transform: Affine geotransform.
        map_obj: Existing ``folium.Map`` to add to. A new map is created
            if ``None``.
        vmin: Min value for colormap normalization.
        vmax: Max value for colormap normalization.
        cmap: Matplotlib colormap name.
        name: Layer name for the layer control.
        opacity: Layer opacity (0–1).
        zoom_start: Initial zoom level if creating a new map.

    Returns:
        ``folium.Map`` with the overlay added.
    """
    folium = _check_folium()
    H, W = data.shape
    bounds = _to_latlon_bounds(crs, transform, H, W)

    rgba = _apply_colormap(data, vmin=vmin, vmax=vmax, cmap=cmap)
    png_bytes = _array_to_png(rgba)

    import base64

    img_b64 = base64.b64encode(png_bytes).decode()
    img_url = f"data:image/png;base64,{img_b64}"

    if map_obj is None:
        center_lat = (bounds[0][0] + bounds[1][0]) / 2
        center_lon = (bounds[0][1] + bounds[1][1]) / 2
        map_obj = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
        )

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=opacity,
        name=name,
        interactive=True,
    ).add_to(map_obj)

    return map_obj


def overlay_rgb(
    rgb: np.ndarray,
    crs: CRS | str | int,
    transform: Affine | tuple,
    *,
    map_obj: Any | None = None,
    name: str = "RGB composite",
    opacity: float = 0.8,
    zoom_start: int = 10,
) -> Any:
    """Overlay a 3-band RGB image on a folium map.

    Args:
        rgb: uint8 array of shape ``(3, H, W)`` (band-first) or ``(H, W, 3)``.
        crs: Coordinate reference system.
        transform: Affine geotransform.
        map_obj: Existing ``folium.Map``. Created if ``None``.
        name: Layer name.
        opacity: Layer opacity.
        zoom_start: Initial zoom level.

    Returns:
        ``folium.Map`` with the overlay added.
    """
    folium = _check_folium()

    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)  # (3,H,W) → (H,W,3)

    H, W, _ = rgb.shape
    bounds = _to_latlon_bounds(crs, transform, H, W)

    # Add alpha channel
    alpha = np.full((H, W, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    png_bytes = _array_to_png(rgba)

    import base64

    img_b64 = base64.b64encode(png_bytes).decode()
    img_url = f"data:image/png;base64,{img_b64}"

    if map_obj is None:
        center_lat = (bounds[0][0] + bounds[1][0]) / 2
        center_lon = (bounds[0][1] + bounds[1][1]) / 2
        map_obj = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
        )

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=opacity,
        name=name,
        interactive=True,
    ).add_to(map_obj)

    return map_obj


def add_layer_control(map_obj: Any) -> Any:
    """Add a layer control toggle to a folium map.

    Args:
        map_obj: ``folium.Map`` instance.

    Returns:
        The same map object.
    """
    folium = _check_folium()
    folium.LayerControl().add_to(map_obj)
    return map_obj
