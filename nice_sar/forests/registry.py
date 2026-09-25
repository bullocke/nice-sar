"""Method and provider registries for forest masking workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ForestMaskMethodSpec:
    """Metadata describing a forest-mask method.

    Attributes:
        name: Stable API identifier.
        description: Short human-readable description.
        implemented: Whether the method is implemented in the current release.
        product_types: NISAR product types or external source categories expected.
        required_inputs: Required data inputs for the method.
        references: Key literature or provenance references.
        tags: Short capability tags for discovery.
        default_threshold_db: Optional default threshold in dB for threshold methods.
    """

    name: str
    description: str
    implemented: bool
    product_types: tuple[str, ...]
    required_inputs: tuple[str, ...]
    references: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    default_threshold_db: float | None = None


@dataclass(frozen=True)
class ForestMaskProviderSpec:
    """Metadata describing an external forest-mask provider."""

    name: str
    description: str
    implemented: bool
    source_types: tuple[str, ...]
    notes: tuple[str, ...] = ()


METHOD_REGISTRY: dict[str, ForestMaskMethodSpec] = {
    "gcov_hv_threshold": ForestMaskMethodSpec(
        name="gcov_hv_threshold",
        description="GCOV HV backscatter thresholding with configurable threshold.",
        implemented=True,
        product_types=("GCOV", "GeoTIFF"),
        required_inputs=("hv_backscatter",),
        references=(
            "Ramachandran et al. (2024)",
            "NISAR and Dixon (2025)",
        ),
        tags=("gcov", "threshold", "hv", "implemented"),
        default_threshold_db=-20.0,
    ),
    "gcov_hv_threshold_ramachandran": ForestMaskMethodSpec(
        name="gcov_hv_threshold_ramachandran",
        description="GCOV HV threshold preset using -20 dB.",
        implemented=True,
        product_types=("GCOV", "GeoTIFF"),
        required_inputs=("hv_backscatter",),
        references=("Ramachandran et al. (2024)",),
        tags=("gcov", "threshold", "preset", "implemented"),
        default_threshold_db=-20.0,
    ),
    "gcov_hv_threshold_jaxa_biomass": ForestMaskMethodSpec(
        name="gcov_hv_threshold_jaxa_biomass",
        description="GCOV HV threshold preset using -13 dB.",
        implemented=True,
        product_types=("GCOV", "GeoTIFF"),
        required_inputs=("hv_backscatter",),
        references=("NISAR and Dixon (2025)",),
        tags=("gcov", "threshold", "preset", "implemented"),
        default_threshold_db=-13.0,
    ),
    "external_raster": ForestMaskMethodSpec(
        name="external_raster",
        description="Load an external raster or COG and interpret it as a forest mask.",
        implemented=True,
        product_types=("GeoTIFF", "COG", "external"),
        required_inputs=("raster_mask",),
        references=("User-provided forest mask raster",),
        tags=("external", "raster", "implemented"),
    ),
    "gcov_dprvi_glcm_texture": ForestMaskMethodSpec(
        name="gcov_dprvi_glcm_texture",
        description="GCOV DpRVI and GLCM texture-based forest discrimination.",
        implemented=False,
        product_types=("GCOV",),
        required_inputs=("hh_backscatter", "hv_backscatter"),
        references=("Tesser et al. (2025)",),
        tags=("gcov", "texture", "planned"),
    ),
    "gslc_coherence_assisted": ForestMaskMethodSpec(
        name="gslc_coherence_assisted",
        description="GSLC or GUNW coherence-assisted forest masking.",
        implemented=False,
        product_types=("GSLC", "GUNW"),
        required_inputs=("coherence", "backscatter"),
        references=("NISAR and Dixon (2025)",),
        tags=("insar", "coherence", "planned"),
    ),
    "hv_timeseries_disturbance": ForestMaskMethodSpec(
        name="hv_timeseries_disturbance",
        description="Multitemporal HV disturbance-based forest masking.",
        implemented=False,
        product_types=("GCOV", "GSLC"),
        required_inputs=("hv_timeseries",),
        references=("Siqueira et al. (2021)",),
        tags=("timeseries", "disturbance", "planned"),
    ),
    "rfdi_multisensor": ForestMaskMethodSpec(
        name="rfdi_multisensor",
        description="RFDI plus external optical or ancillary products.",
        implemented=False,
        product_types=("GCOV", "external"),
        required_inputs=("hh_backscatter", "hv_backscatter", "ancillary_mask"),
        references=("Flores-Anderson et al. (2025)",),
        tags=("rfdi", "multisensor", "planned"),
    ),
    "nisar_official_mask": ForestMaskMethodSpec(
        name="nisar_official_mask",
        description="Official NISAR forest-related mask ingestion.",
        implemented=False,
        product_types=("NISAR_L3", "external"),
        required_inputs=("provider_product",),
        references=("NISAR and Dixon (2025)",),
        tags=("official", "external", "planned"),
    ),
}


PROVIDER_REGISTRY: dict[str, ForestMaskProviderSpec] = {
    "external_raster": ForestMaskProviderSpec(
        name="external_raster",
        description="Generic local, HTTP, or S3 GeoTIFF/COG forest mask source.",
        implemented=True,
        source_types=("local", "http", "https", "s3"),
        notes=("Cloud access depends on GDAL/rasterio transport support.",),
    ),
    "umd_gfw_tree_cover": ForestMaskProviderSpec(
        name="umd_gfw_tree_cover",
        description="UMD or Global Forest Watch tree-cover style products.",
        implemented=False,
        source_types=("http", "https", "s3"),
        notes=("Provider-specific URL resolution is planned, not yet automated.",),
    ),
    "nisar_official_mask": ForestMaskProviderSpec(
        name="nisar_official_mask",
        description="Official NISAR forest-related external products or validation masks.",
        implemented=False,
        source_types=("http", "https", "s3"),
        notes=("Provider-specific catalog integration is planned, not yet automated.",),
    ),
}


def list_method_specs(include_unimplemented: bool = True) -> list[dict[str, object]]:
    """Return registered method metadata as dictionaries."""
    specs = METHOD_REGISTRY.values()
    if not include_unimplemented:
        specs = [spec for spec in specs if spec.implemented]
    return [asdict(spec) for spec in sorted(specs, key=lambda item: item.name)]


def list_provider_specs(include_unimplemented: bool = True) -> list[dict[str, object]]:
    """Return registered provider metadata as dictionaries."""
    specs = PROVIDER_REGISTRY.values()
    if not include_unimplemented:
        specs = [spec for spec in specs if spec.implemented]
    return [asdict(spec) for spec in sorted(specs, key=lambda item: item.name)]


def get_method_spec(name: str) -> ForestMaskMethodSpec:
    """Return a method specification or raise a helpful error."""
    if name not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown forest-mask method: {name!r}. Available methods: {available}")
    return METHOD_REGISTRY[name]


def get_provider_spec(name: str) -> ForestMaskProviderSpec:
    """Return a provider specification or raise a helpful error."""
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown forest-mask provider: {name!r}. Available providers: {available}")
    return PROVIDER_REGISTRY[name]