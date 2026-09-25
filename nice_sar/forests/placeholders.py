"""Placeholder implementations for planned forest mask methods."""

from __future__ import annotations


def gcov_dprvi_glcm_texture(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for GCOV texture-based forest masking."""
    raise NotImplementedError(
        "Method 'gcov_dprvi_glcm_texture' is planned but not implemented. "
        "Expected inputs include dual-pol GCOV backscatter and derived texture layers."
    )


def gslc_coherence_assisted(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for coherence-assisted forest masking."""
    raise NotImplementedError(
        "Method 'gslc_coherence_assisted' is planned but not implemented. "
        "Expected inputs include GSLC or GUNW coherence plus supporting backscatter layers."
    )


def hv_timeseries_disturbance(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for multitemporal HV disturbance masking."""
    raise NotImplementedError(
        "Method 'hv_timeseries_disturbance' is planned but not implemented. "
        "Expected inputs include multitemporal HV backscatter stacks and disturbance logic."
    )


def rfdi_multisensor(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for RFDI plus multisensor masking."""
    raise NotImplementedError(
        "Method 'rfdi_multisensor' is planned but not implemented. "
        "Expected inputs include HH/HV backscatter and ancillary optical or land-cover data."
    )


def nisar_official_mask(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for official NISAR forest-mask ingestion."""
    raise NotImplementedError(
        "Method 'nisar_official_mask' is planned but not implemented. "
        "Provider-specific product resolution and ingestion remain future work."
    )