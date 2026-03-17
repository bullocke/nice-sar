"""RGB composite generation from SAR polarimetric data.

Contains 12 composite algorithms for dual-pol (HH/HV) NISAR data,
each targeting different visualization goals. Includes a dispatcher
function :func:`make_rgb` for convenient access by name.

All composite functions accept numpy arrays in linear power units
(unless noted otherwise) and return ``(rgb_stack, band_names)`` where
``rgb_stack`` is shape ``(3, H, W)`` uint8 and ``band_names`` is a list
of 3 description strings.
"""

from __future__ import annotations

import numpy as np

from nice_sar.viz.display import gamma_correct, percentile_stretch, to_uint8

# Type alias for composite return
RGBResult = tuple[np.ndarray, list[str]]


def rgb_standard_dualpol(
    hh_db: np.ndarray, hv_db: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Standard dual-pol RGB (reference baseline).

    R=HH_dB, G=HV_dB, B=HH_dB-HV_dB. Vegetation appears yellow.

    Args:
        hh_db: HH backscatter in dB.
        hv_db: HV backscatter in dB.
        gamma: Gamma correction exponent.
    """
    R = gamma_correct(percentile_stretch(hh_db), gamma)
    G = gamma_correct(percentile_stretch(hv_db), gamma)
    B = gamma_correct(percentile_stretch(hh_db - hv_db), gamma)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=HH_dB", "G=HV_dB", "B=HH-HV_dB"],
    )


def rgb_vegetation_green(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.45
) -> RGBResult:
    """Modified scheme to make vegetation GREEN.

    R = HH/(HH+2*HV), G = HV/(HH+HV), B = sqrt(SPAN).

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    total = hh_linear + hv_linear + eps
    R_raw = hh_linear / (hh_linear + 2 * hv_linear + eps)
    G_raw = hv_linear / total
    B_raw = np.sqrt(total)

    R = gamma_correct(percentile_stretch(R_raw, 1, 99), gamma)
    G = gamma_correct(percentile_stretch(G_raw, 1, 99), gamma)
    B = gamma_correct(percentile_stretch(B_raw, 2, 98), gamma * 1.2)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=HH/(HH+2HV)", "G=HV/(HH+HV)", "B=sqrt(SPAN)"],
    )


def rgb_natural_color_v1(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.4
) -> RGBResult:
    """Natural color attempt using intensity-ratio mapping.

    Uses HV/HH ratio for hue determination and SPAN for brightness.
    Target: vegetation=green, urban=gray, water=dark, bare=brown.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    ratio = hv_linear / (hh_linear + eps)
    vol_proxy = hv_linear / span

    ratio_stretched = percentile_stretch(ratio, 5, 95)
    intensity = gamma_correct(percentile_stretch(np.sqrt(span), 2, 98), gamma)

    G = intensity * (0.3 + 0.7 * gamma_correct(ratio_stretched, 0.6))
    R = intensity * (0.3 + 0.5 * (1 - gamma_correct(vol_proxy, 0.5)))
    low_scatter = 1 - percentile_stretch(span, 5, 50)
    B = intensity * (0.25 + 0.3 * low_scatter)

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=Intensity*(surface)", "G=Intensity*(volume)", "B=Intensity*(low-scatter)"],
    )


def rgb_natural_color_v2(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.45
) -> RGBResult:
    """Natural color v2 — suppresses red where HV is high (vegetation).

    R=HH*(1-vol_proxy), G=HV, B=sqrt(HH*HV).

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span

    R_raw = hh_linear * (1 - vol_proxy)
    G_raw = hv_linear
    B_raw = np.sqrt(hh_linear * hv_linear + eps)

    R = gamma_correct(percentile_stretch(R_raw, 2, 98), gamma)
    G = gamma_correct(percentile_stretch(G_raw, 2, 98), gamma)
    B = gamma_correct(percentile_stretch(B_raw, 2, 98), gamma)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=HH*(1-volproxy)", "G=HV", "B=sqrt(HH*HV)"],
    )


def rgb_forest_enhanced(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Forest-enhanced composite for forest/non-forest mapping.

    R=RFDI, G=volume proxy, B=sqrt(SPAN).

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    rfdi = (hh_linear - hv_linear) / span
    vol_proxy = hv_linear / span
    rfdi_norm = (rfdi + 1) / 2

    R = gamma_correct(percentile_stretch(rfdi_norm, 2, 98), gamma)
    G = gamma_correct(percentile_stretch(vol_proxy, 2, 98), gamma * 0.9)
    B = gamma_correct(percentile_stretch(np.sqrt(span), 2, 98), gamma)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=RFDI", "G=VolProxy", "B=sqrt(SPAN)"],
    )


def rgb_landcover_intuitive(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.45
) -> RGBResult:
    """Land cover intuitive RGB targeting natural color expectations.

    Forest=dark-green, urban=gray/white, water=dark, bare=brown/tan.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span
    surf_proxy = hh_linear / span
    intensity = np.sqrt(span)

    vol_n = percentile_stretch(vol_proxy, 5, 95)
    surf_n = percentile_stretch(surf_proxy, 5, 95)
    int_n = gamma_correct(percentile_stretch(intensity, 2, 98), gamma)

    G = int_n * (0.4 + 0.6 * vol_n)
    urban_indicator = surf_n * int_n
    R = int_n * (0.4 + 0.4 * surf_n) + 0.2 * urban_indicator
    B = int_n * 0.35

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=surface-weighted", "G=volume-weighted", "B=intensity-base"],
    )


def rgb_pseudo_natural(
    hh_db: np.ndarray, hv_db: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Pseudo-natural using dB values with modified channel assignment.

    R=HH_dB, G=2*HV_dB-HH_dB, B=(HH_dB+HV_dB)/2.

    Args:
        hh_db: HH backscatter in dB.
        hv_db: HV backscatter in dB.
        gamma: Gamma correction exponent.
    """
    R_raw = hh_db
    G_raw = 2 * hv_db - hh_db
    B_raw = (hh_db + hv_db) / 2

    R = gamma_correct(percentile_stretch(R_raw, 2, 98), gamma)
    G = gamma_correct(percentile_stretch(G_raw, 2, 98), gamma)
    B = gamma_correct(percentile_stretch(B_raw, 2, 98), gamma)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=HH_dB", "G=2*HV-HH_dB", "B=avg_dB"],
    )


def rgb_enhanced_contrast(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.4
) -> RGBResult:
    """Enhanced contrast RGB using log-ratio coloring.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    hh_log = np.log10(hh_linear + eps)
    hv_log = np.log10(hv_linear + eps)
    diff = hh_log - hv_log
    avg = (hh_log + hv_log) / 2
    pol = -diff

    R = gamma_correct(percentile_stretch(diff, 5, 95), gamma)
    G = gamma_correct(percentile_stretch(pol, 5, 95), gamma)
    B = gamma_correct(percentile_stretch(avg, 2, 98), gamma)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=log(HH/HV)", "G=-log(HH/HV)", "B=log(avg)"],
    )


def rgb_natural_v1_improved(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Improved Natural V1 — darkens forests for optical-like appearance.

    Inverts SAR brightness so forests appear darker (as in optical imagery).

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span
    intensity = np.sqrt(span)

    vol_n = percentile_stretch(vol_proxy, 5, 95)
    int_n = percentile_stretch(intensity, 2, 98)

    brightness_modifier = 1.0 - 0.4 * vol_n
    base_intensity = gamma_correct(int_n, gamma) * brightness_modifier

    G = base_intensity * (0.5 + 0.5 * vol_n)
    bare_indicator = 1 - vol_n
    R = base_intensity * (0.4 + 0.5 * bare_indicator)
    B = base_intensity * (0.3 + 0.2 * bare_indicator)

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=intensity*bare", "G=intensity*veg", "B=intensity*base"],
    )


def rgb_natural_v1_improved_b(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.55
) -> RGBResult:
    """Alternative improved Natural V1 — aggressive forest darkening with saturation boost.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span
    vol_n = percentile_stretch(vol_proxy, 3, 97)

    int_n = gamma_correct(percentile_stretch(np.sqrt(span), 2, 98), gamma)
    forest_darkness = 0.4 + 0.6 * (1 - vol_n) ** 1.5
    adj_intensity = int_n * forest_darkness
    veg_saturation = 0.3 + 0.7 * vol_n

    G = adj_intensity * (0.6 + 0.4 * veg_saturation)
    R = adj_intensity * (0.7 - 0.3 * veg_saturation)
    B = adj_intensity * (0.4 - 0.15 * veg_saturation)

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=adj_int*(1-veg)", "G=adj_int*veg", "B=adj_int*base"],
    )


def rgb_natural_v2_improved(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Improved Natural V2 — maps forests to dark green, bare to tan.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span
    vol_n = percentile_stretch(vol_proxy, 5, 95)
    int_n = gamma_correct(percentile_stretch(np.sqrt(span), 2, 98), gamma)

    darkness = 0.5 + 0.5 * (1 - vol_n)
    adj_int = int_n * darkness

    R = adj_int * (0.6 - 0.4 * vol_n)
    G = adj_int * (0.4 + 0.15 * vol_n)
    B = adj_int * (0.25 - 0.1 * vol_n)

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=tan_to_green", "G=veg_weighted", "B=warm_tint"],
    )


def rgb_optical_mimic(
    hh_linear: np.ndarray, hv_linear: np.ndarray, gamma: float = 0.5
) -> RGBResult:
    """Mimics optical satellite imagery appearance.

    Blends between dark-green (forest) and tan (bare) color targets,
    with brightness modulated to simulate canopy shadow.

    Args:
        hh_linear: HH backscatter in linear power.
        hv_linear: HV backscatter in linear power.
        gamma: Gamma correction exponent.
    """
    eps = 1e-10
    span = hh_linear + hv_linear + eps
    vol_proxy = hv_linear / span
    vol_n = percentile_stretch(vol_proxy, 5, 95)

    raw_int = np.sqrt(span)
    int_n = percentile_stretch(raw_int, 2, 98)
    optical_int = int_n * (0.4 + 0.6 * (1 - vol_n) ** 1.2)
    optical_int = gamma_correct(optical_int, gamma)

    # Color targets
    forest_r, forest_g, forest_b = 0.20, 0.45, 0.15
    bare_r, bare_g, bare_b = 0.70, 0.60, 0.45
    blend = vol_n

    R = optical_int * (bare_r + (forest_r - bare_r) * blend)
    G = optical_int * (bare_g + (forest_g - bare_g) * blend)
    B = optical_int * (bare_b + (forest_b - bare_b) * blend)

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    return (
        np.stack([to_uint8(R), to_uint8(G), to_uint8(B)], axis=0),
        ["R=optical_blend", "G=optical_blend", "B=optical_blend"],
    )


def make_dualpol_rgb(
    hh_db: np.ndarray, hv_db: np.ndarray
) -> np.ndarray:
    """Build a dual-pol RGB stack in float32 (pre-stretch).

    R=HH_dB, G=HV_dB, B=HH_dB-HV_dB.

    Args:
        hh_db: HH backscatter in dB.
        hv_db: HV backscatter in dB.

    Returns:
        Array of shape ``(3, H, W)`` in float32.
    """
    R = hh_db.copy()
    G = hv_db.copy()
    B = hh_db - hv_db
    return np.asarray(np.stack([R, G, B], axis=0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Registry of available RGB methods and their input requirements.
_RGB_METHODS: dict[str, dict] = {
    "standard": {"func": rgb_standard_dualpol, "input": "db"},
    "vegetation_green": {"func": rgb_vegetation_green, "input": "linear"},
    "natural_v1": {"func": rgb_natural_color_v1, "input": "linear"},
    "natural_v2": {"func": rgb_natural_color_v2, "input": "linear"},
    "forest_enhanced": {"func": rgb_forest_enhanced, "input": "linear"},
    "landcover": {"func": rgb_landcover_intuitive, "input": "linear"},
    "pseudo_natural": {"func": rgb_pseudo_natural, "input": "db"},
    "enhanced_contrast": {"func": rgb_enhanced_contrast, "input": "linear"},
    "natural_v1_improved": {"func": rgb_natural_v1_improved, "input": "linear"},
    "natural_v1_improved_b": {"func": rgb_natural_v1_improved_b, "input": "linear"},
    "natural_v2_improved": {"func": rgb_natural_v2_improved, "input": "linear"},
    "optical_mimic": {"func": rgb_optical_mimic, "input": "linear"},
}


def make_rgb(
    hh: np.ndarray,
    hv: np.ndarray,
    method: str = "vegetation_green",
    **kwargs,
) -> RGBResult:
    """Generate an RGB composite using a named method.

    Args:
        hh: HH backscatter array (linear or dB, depending on method).
        hv: HV backscatter array (same scale as hh).
        method: Name of the composite method. See ``list_rgb_methods()``.
        **kwargs: Additional keyword arguments passed to the composite function
            (e.g., ``gamma``).

    Returns:
        Tuple of ``(rgb_stack, band_names)`` where ``rgb_stack`` is
        ``(3, H, W)`` uint8.

    Raises:
        ValueError: If method name is not recognized.
    """
    if method not in _RGB_METHODS:
        available = ", ".join(sorted(_RGB_METHODS.keys()))
        raise ValueError(f"Unknown RGB method '{method}'. Available: {available}")
    result: RGBResult = _RGB_METHODS[method]["func"](hh, hv, **kwargs)
    return result


def list_rgb_methods() -> list[str]:
    """Return list of available RGB composite method names."""
    return sorted(_RGB_METHODS.keys())
