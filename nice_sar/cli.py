"""Command-line interface for nice-sar.

Provides subcommands for common NISAR data workflows:
- ``info``: Print product metadata
- ``read``: Read and export product bands
- ``multilook``: Apply spatial multilooking
- ``rgb``: Generate RGB composites
- ``insar``: InSAR processing utilities
- ``timeseries``: Time series change detection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> None:
    """Print product metadata as JSON."""
    from nice_sar.io.hdf5 import get_frequencies, get_polarizations, open_nisar
    from nice_sar.io.products import read_identification

    h5 = open_nisar(args.input)
    try:
        meta = read_identification(h5)
        freqs = get_frequencies(h5)
        pols = {}
        for f in freqs:
            try:
                pols[f] = get_polarizations(h5, f)
            except KeyError:
                pols[f] = []
        meta["frequencies"] = freqs
        meta["polarizations"] = pols
    finally:
        h5.close()

    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def cmd_read(args: argparse.Namespace) -> None:
    """Read a product band and export to GeoTIFF."""
    from nice_sar.io.geotiff import write_geotiff
    from nice_sar.io.products import (
        read_gcov,
        read_goff,
        read_gslc,
        read_gunw,
        read_rslc,
    )

    readers = {
        "GCOV": read_gcov,
        "GSLC": read_gslc,
        "GUNW": read_gunw,
        "GOFF": read_goff,
        "RSLC": read_rslc,
    }
    product = args.product.upper()
    reader = readers.get(product)
    if reader is None:
        logger.error("Unknown product type: %s", product)
        sys.exit(1)

    kwargs: dict = {
        "source": args.input,
        "frequency": args.frequency,
        "polarization": args.polarization,
    }
    if product in ("GUNW", "GOFF") and args.layer:
        kwargs["layer"] = args.layer

    da_xr = reader(**kwargs)
    data = da_xr.values

    if np.iscomplexobj(data):
        # Export amplitude for complex data
        data = np.abs(data).astype(np.float32)
        logger.info("Complex data — exporting amplitude.")

    from pyproj import CRS
    from rasterio.transform import Affine

    if "crs" in da_xr.attrs and "transform" in da_xr.attrs:
        crs = CRS.from_user_input(da_xr.attrs["crs"])
        transform = Affine(*da_xr.attrs["transform"])
        write_geotiff(args.output, data, crs=crs, transform=transform)
    else:
        # RSLC — no geo info, write raw
        write_geotiff(args.output, data)

    logger.info("Wrote %s", args.output)


# ---------------------------------------------------------------------------
# multilook
# ---------------------------------------------------------------------------


def cmd_multilook(args: argparse.Namespace) -> None:
    """Apply multilooking to a product band and export."""
    from nice_sar.io.geotiff import write_geotiff
    from nice_sar.io.products import read_gcov
    from nice_sar.preprocess.multilook import multilook

    da_xr = read_gcov(args.input, frequency=args.frequency, polarization=args.polarization)
    data = da_xr.values
    result = multilook(data, looks_y=args.looks_y, looks_x=args.looks_x)

    from pyproj import CRS
    from rasterio.transform import Affine

    crs = CRS.from_user_input(da_xr.attrs["crs"])
    old_t = Affine(*da_xr.attrs["transform"])
    new_t = Affine(
        old_t.a * args.looks_x, old_t.b, old_t.c,
        old_t.d, old_t.e * args.looks_y, old_t.f,
    )
    write_geotiff(args.output, result, crs=crs, transform=new_t)
    logger.info("Wrote multilooked output: %s", args.output)


# ---------------------------------------------------------------------------
# rgb
# ---------------------------------------------------------------------------


def cmd_rgb(args: argparse.Namespace) -> None:
    """Generate an RGB composite and export as GeoTIFF."""
    from nice_sar.io.geotiff import write_geotiff
    from nice_sar.io.products import read_gcov
    from nice_sar.preprocess.calibration import linear_to_db
    from nice_sar.viz.rgb import make_rgb

    hh_xr = read_gcov(args.input, frequency=args.frequency, polarization="HH")
    hv_xr = read_gcov(args.input, frequency=args.frequency, polarization="HV")

    hh = hh_xr.values
    hv = hv_xr.values

    if args.composite in ("standard_dualpol",):
        hh_in = linear_to_db(hh)
        hv_in = linear_to_db(hv)
    else:
        hh_in = hh
        hv_in = hv

    rgb, _ = make_rgb(args.composite, hh_in, hv_in)

    from pyproj import CRS
    from rasterio.transform import Affine

    crs = CRS.from_user_input(hh_xr.attrs["crs"])
    transform = Affine(*hh_xr.attrs["transform"])
    write_geotiff(args.output, rgb, crs=crs, transform=transform, count=3)
    logger.info("Wrote RGB composite: %s", args.output)


# ---------------------------------------------------------------------------
# insar
# ---------------------------------------------------------------------------


def cmd_insar(args: argparse.Namespace) -> None:
    """InSAR processing: form interferogram or convert GUNW phase."""
    if args.subcommand == "phase2disp":
        from nice_sar.analysis.insar import (
            apply_ionospheric_correction,
            phase_to_displacement,
        )
        from nice_sar.io.geotiff import write_geotiff
        from nice_sar.io.products import read_gunw

        phase_xr = read_gunw(
            args.input, polarization=args.polarization, layer="unwrappedPhase"
        )
        phase = phase_xr.values

        if args.apply_iono:
            iono_xr = read_gunw(
                args.input, polarization=args.polarization, layer="ionospherePhaseScreen"
            )
            phase = apply_ionospheric_correction(phase, iono_xr.values)

        disp = phase_to_displacement(phase, wavelength=args.wavelength)

        from pyproj import CRS
        from rasterio.transform import Affine

        crs = CRS.from_user_input(phase_xr.attrs["crs"])
        transform = Affine(*phase_xr.attrs["transform"])
        write_geotiff(args.output, disp, crs=crs, transform=transform)
        logger.info("Wrote displacement: %s", args.output)

    elif args.subcommand == "coherence":
        from nice_sar.io.geotiff import write_geotiff
        from nice_sar.io.products import read_gunw

        coh_xr = read_gunw(
            args.input,
            polarization=args.polarization,
            layer="coherenceMagnitude",
            posting=args.posting,
        )
        from pyproj import CRS
        from rasterio.transform import Affine

        crs = CRS.from_user_input(coh_xr.attrs["crs"])
        transform = Affine(*coh_xr.attrs["transform"])
        write_geotiff(args.output, coh_xr.values, crs=crs, transform=transform)
        logger.info("Wrote coherence: %s", args.output)


# ---------------------------------------------------------------------------
# timeseries
# ---------------------------------------------------------------------------


def cmd_timeseries(args: argparse.Namespace) -> None:
    """Time series analysis subcommands."""
    from nice_sar.io.products import read_gcov

    # Build stack from list of files
    files = [Path(f) for f in args.inputs]
    arrays = []
    attrs = None
    for f in files:
        da_xr = read_gcov(f, frequency=args.frequency, polarization=args.polarization)
        arrays.append(da_xr.values)
        if attrs is None:
            attrs = da_xr.attrs

    stack = np.stack(arrays, axis=0)

    from nice_sar.io.geotiff import write_geotiff

    if args.method == "cov":
        from nice_sar.analysis.timeseries import coefficient_of_variation

        result = coefficient_of_variation(stack)
    elif args.method == "cusum":
        from nice_sar.analysis.timeseries import cusum

        cs_result = cusum(stack, threshold=args.threshold)
        result = cs_result.magnitude
    else:
        logger.error("Unknown method: %s", args.method)
        sys.exit(1)

    if attrs and "crs" in attrs:
        from pyproj import CRS
        from rasterio.transform import Affine

        crs = CRS.from_user_input(attrs["crs"])
        transform = Affine(*attrs["transform"])
        write_geotiff(args.output, result, crs=crs, transform=transform)
    else:
        write_geotiff(args.output, result)

    logger.info("Wrote %s result: %s", args.method, args.output)


# ---------------------------------------------------------------------------
# forests
# ---------------------------------------------------------------------------


def cmd_forests(args: argparse.Namespace) -> None:
    """Forest masking utilities."""
    from nice_sar.forests import generate_forest_mask, list_forest_mask_methods
    from nice_sar.io.geotiff import write_geotiff

    if args.subcommand == "list-methods":
        methods = list_forest_mask_methods(include_unimplemented=args.include_unimplemented)
        print(json.dumps(methods, indent=2))
        return

    result = generate_forest_mask(
        source=args.input,
        method=args.method,
        frequency=args.frequency,
        polarization=args.polarization,
        units=args.units,
        threshold_db=args.threshold_db,
        target=args.target,
        band=args.band,
        threshold=args.threshold,
        mask_values=tuple(args.mask_value) if args.mask_value else None,
        invert=args.invert,
        resampling=args.resampling,
    )

    if hasattr(result.mask, "values"):
        mask_data = np.asarray(result.mask.values, dtype=np.float32)
        attrs = result.mask.attrs
    else:
        mask_data = np.asarray(result.mask, dtype=np.float32)
        attrs = {}

    if "crs" not in attrs or "transform" not in attrs:
        logger.error("Forest mask output is missing georeferencing metadata required for GeoTIFF export.")
        sys.exit(1)

    from pyproj import CRS
    from rasterio.transform import Affine

    mask_data = np.where(np.isfinite(mask_data), mask_data.astype(np.float32), np.nan)
    write_geotiff(
        args.output,
        mask_data,
        crs=CRS.from_user_input(attrs["crs"]),
        transform=Affine(*attrs["transform"]),
        description=f"forest_mask:{result.method}",
    )
    logger.info("Wrote forest mask: %s", args.output)

    if args.confidence_output and result.confidence is not None:
        if hasattr(result.confidence, "values"):
            confidence = np.asarray(result.confidence.values, dtype=np.float32)
            conf_attrs = result.confidence.attrs
        else:
            confidence = np.asarray(result.confidence, dtype=np.float32)
            conf_attrs = attrs
        write_geotiff(
            args.confidence_output,
            confidence,
            crs=CRS.from_user_input(conf_attrs["crs"]),
            transform=Affine(*conf_attrs["transform"]),
            description=f"forest_mask_confidence:{result.method}",
        )
        logger.info("Wrote forest mask confidence: %s", args.confidence_output)


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------


def cmd_subset(args: argparse.Namespace) -> None:
    """Download a spatial subset of a NISAR product as GeoTIFF."""
    from nice_sar.auth.earthdata import (
        get_https_filesystem,
        get_s3_filesystem,
        login,
    )
    from nice_sar.io.bbox_parser import parse_bbox
    from nice_sar.io.subset import subset_product
    from nice_sar.search.asf import search_nisar

    # Resolve bbox from --bbox or --bbox-file
    bbox = parse_bbox(str(args.bbox_file)) if args.bbox_file else parse_bbox(args.bbox)

    # Authenticate
    login()
    import os

    if os.environ.get("AWS_DEFAULT_REGION") == "us-west-2":
        fs = get_s3_filesystem()
        access = "s3"
    else:
        fs = get_https_filesystem()
        access = "https"

    # Search
    results = search_nisar(
        product_type=args.product.upper(),
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        start=args.start,
        end=args.end,
        max_results=args.max_granules,
    )
    if not results:
        logger.error("No granules found for the specified parameters.")
        sys.exit(1)

    logger.info("Found %d granule(s), subsetting up to %d", len(results), args.max_granules)

    all_outputs: list[Path] = []
    for i, granule in enumerate(results[: args.max_granules]):
        # Extract URL from asf_search result object
        if access == "s3":
            s3_urls = granule.properties.get("s3Urls", [])
            h5_urls = [u for u in s3_urls if u.endswith(".h5")]
            url = h5_urls[0] if h5_urls else granule.properties["url"]
        else:
            url = granule.properties["url"]
        logger.info("Granule %d/%d: %s", i + 1, min(len(results), args.max_granules), url)

        pols = args.polarization if args.polarization else None
        layers = [args.layer] if args.layer else None

        outputs = subset_product(
            source=url,
            product=args.product.upper(),
            bbox=bbox,
            frequency=args.frequency,
            polarizations=pols,
            layers=layers,
            output_dir=args.output_dir,
            filesystem=fs,
            confirm=not args.no_confirm,
        )
        all_outputs.extend(outputs)

    if all_outputs:
        print(f"\nDone — {len(all_outputs)} file(s) written to {args.output_dir}/")
        for p in all_outputs:
            print(f"  {p}")
    else:
        print("No files were written.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the nice-sar CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nice-sar",
        description="NISAR SAR data processing toolkit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- info ---
    p_info = sub.add_parser("info", help="Print product metadata")
    p_info.add_argument("input", type=Path, help="Path to NISAR HDF5 file")

    # --- read ---
    p_read = sub.add_parser("read", help="Read & export a product band")
    p_read.add_argument("input", type=Path, help="Path to NISAR HDF5 file")
    p_read.add_argument("output", type=Path, help="Output GeoTIFF path")
    p_read.add_argument("-p", "--product", required=True, help="Product type (GCOV/GSLC/GUNW/GOFF/RSLC)")
    p_read.add_argument("--frequency", default="A", help="Frequency (A/B)")
    p_read.add_argument("--polarization", default="HH", help="Polarization (HH/HV/VV)")
    p_read.add_argument("--layer", default=None, help="Layer name (GUNW/GOFF)")

    # --- multilook ---
    p_ml = sub.add_parser("multilook", help="Apply multilooking")
    p_ml.add_argument("input", type=Path, help="Path to NISAR GCOV HDF5 file")
    p_ml.add_argument("output", type=Path, help="Output GeoTIFF path")
    p_ml.add_argument("--frequency", default="A")
    p_ml.add_argument("--polarization", default="HH")
    p_ml.add_argument("--looks-y", type=int, default=3, help="Y looks")
    p_ml.add_argument("--looks-x", type=int, default=3, help="X looks")

    # --- rgb ---
    p_rgb = sub.add_parser("rgb", help="Generate RGB composite")
    p_rgb.add_argument("input", type=Path, help="Path to NISAR GCOV HDF5 file")
    p_rgb.add_argument("output", type=Path, help="Output GeoTIFF path")
    p_rgb.add_argument("--frequency", default="A")
    p_rgb.add_argument("--composite", default="standard_dualpol", help="Composite name")

    # --- insar ---
    p_insar = sub.add_parser("insar", help="InSAR utilities")
    insar_sub = p_insar.add_subparsers(dest="subcommand", required=True)

    p_phase = insar_sub.add_parser("phase2disp", help="Phase to displacement")
    p_phase.add_argument("input", type=Path, help="GUNW HDF5 file")
    p_phase.add_argument("output", type=Path, help="Output GeoTIFF")
    p_phase.add_argument("--polarization", default="HH")
    p_phase.add_argument("--wavelength", type=float, default=0.24)
    p_phase.add_argument("--apply-iono", action="store_true", help="Apply iono correction")

    p_coh = insar_sub.add_parser("coherence", help="Export coherence")
    p_coh.add_argument("input", type=Path, help="GUNW HDF5 file")
    p_coh.add_argument("output", type=Path, help="Output GeoTIFF")
    p_coh.add_argument("--polarization", default="HH")
    p_coh.add_argument(
        "--posting",
        type=int,
        default=80,
        choices=[20, 80],
        help="Grid posting in metres (default: 80)",
    )

    # --- timeseries ---
    p_ts = sub.add_parser("timeseries", help="Time series analysis")
    p_ts.add_argument("inputs", nargs="+", type=Path, help="GCOV HDF5 files")
    p_ts.add_argument("-o", "--output", type=Path, required=True)
    p_ts.add_argument("--frequency", default="A")
    p_ts.add_argument("--polarization", default="HH")
    p_ts.add_argument("--method", choices=["cov", "cusum"], default="cov")
    p_ts.add_argument("--threshold", type=float, default=None, help="CUSUM threshold")

    # --- forests ---
    p_forest = sub.add_parser("forests", help="Forest masking utilities")
    forest_sub = p_forest.add_subparsers(dest="subcommand", required=True)

    p_forest_list = forest_sub.add_parser("list-methods", help="List available forest mask methods")
    p_forest_list.add_argument(
        "--implemented-only",
        action="store_true",
        help="Show only implemented methods",
    )

    p_forest_generate = forest_sub.add_parser("generate", help="Generate a forest mask")
    p_forest_generate.add_argument("input", type=Path, help="GCOV HDF5 path or external raster path")
    p_forest_generate.add_argument("output", type=Path, help="Output mask GeoTIFF path")
    p_forest_generate.add_argument(
        "--method",
        default="gcov_hv_threshold",
        help="Forest mask method identifier",
    )
    p_forest_generate.add_argument("--frequency", default="A", help="Frequency (A/B)")
    p_forest_generate.add_argument("--polarization", default="HV", help="Polarization to read")
    p_forest_generate.add_argument("--units", default=None, help="Input backscatter units")
    p_forest_generate.add_argument(
        "--threshold-db",
        type=float,
        default=None,
        help="Threshold override for GCOV threshold methods",
    )
    p_forest_generate.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional target grid for external raster alignment",
    )
    p_forest_generate.add_argument("--band", type=int, default=1, help="External raster band")
    p_forest_generate.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for external raster mask generation",
    )
    p_forest_generate.add_argument(
        "--mask-value",
        type=float,
        action="append",
        default=None,
        help="Discrete external raster values treated as forest (repeatable)",
    )
    p_forest_generate.add_argument(
        "--invert",
        action="store_true",
        help="Invert the external raster mask",
    )
    p_forest_generate.add_argument(
        "--resampling",
        default="nearest",
        help="Resampling method for external raster alignment",
    )
    p_forest_generate.add_argument(
        "--confidence-output",
        type=Path,
        default=None,
        help="Optional output path for confidence or threshold-margin raster",
    )

    # --- subset ---
    p_sub = sub.add_parser(
        "subset",
        help="Download a spatial subset of a NISAR product as GeoTIFF",
    )
    bbox_group = p_sub.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument(
        "--bbox",
        help='Bounding box as "west,south,east,north" in WGS84 degrees',
    )
    bbox_group.add_argument(
        "--bbox-file",
        type=Path,
        help="Path to a spatial file (.geojson, .shp, .gpkg) defining the AOI",
    )
    p_sub.add_argument(
        "--product",
        required=True,
        help="Product type (GCOV, GSLC, GUNW, GOFF)",
    )
    p_sub.add_argument("--frequency", default="A", help="Frequency (A/B)")
    p_sub.add_argument(
        "--polarization",
        action="append",
        default=None,
        help="Polarization to download (repeatable, e.g. --polarization HH --polarization HV)",
    )
    p_sub.add_argument("--layer", default=None, help="Layer name (GUNW/GOFF)")
    p_sub.add_argument("--start", default=None, help="Start date (ISO, e.g. 2025-01-01)")
    p_sub.add_argument("--end", default=None, help="End date (ISO, e.g. 2025-06-01)")
    p_sub.add_argument("--max-granules", type=int, default=1, help="Max granules to download")
    p_sub.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("nisar_subset"),
        help="Output directory (default: ./nisar_subset/)",
    )
    p_sub.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip interactive confirmation prompt",
    )

    parser.set_defaults(include_unimplemented=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the nice-sar CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    dispatch = {
        "forests": cmd_forests,
        "info": cmd_info,
        "read": cmd_read,
        "multilook": cmd_multilook,
        "rgb": cmd_rgb,
        "insar": cmd_insar,
        "timeseries": cmd_timeseries,
        "subset": cmd_subset,
    }
    if args.command == "forests" and getattr(args, "subcommand", None) == "list-methods":
        args.include_unimplemented = not args.implemented_only
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
