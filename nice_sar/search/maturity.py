"""NISAR data maturity, collection names, and granule-name parsing.

NISAR products are archived at ASF in separate collections per data maturity:

- ``BETA``: pre-calibration products released February 2026 (acquisitions
  October 2025 - January 2026, CRID lower than ``P05023``).
- ``PROVISIONAL``: calibrated, partially validated products released July 2026
  (CRID ``P05023`` or higher). Forward processing covers acquisitions from
  2026-06-17 onward; a limited set of supplemental frames extends back to
  October 2025.
- ``VALIDATED``: fully validated products (reprocessing expected late 2026).

Collection short names follow ``NISAR_{LEVEL}_{PRODUCT}_{MATURITY}_V1``, with
validated collections omitting the maturity token (``NISAR_L2_GCOV_V1``).

Public API:

- :data:`Maturity` — Accepted maturity values
- :func:`nisar_short_names` — Collection short names for a product and maturity
- :func:`maturity_from_collection` — Maturity implied by a collection short name
- :func:`maturity_from_crid` — Maturity implied by a Composite Release ID
- :class:`NisarGranuleName` / :func:`parse_granule_name` — Parse granule IDs
- :func:`granule_url` — ASF HTTPS download URL for a granule ID
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

Maturity = Literal["provisional", "beta", "validated", "any"]
MATURITIES: tuple[str, ...] = get_args(Maturity)

#: First CRID of the PROVISIONAL (calibrated) release.
PROVISIONAL_MIN_CRID = "P05023"

#: Product type -> processing level used in collection short names.
PRODUCT_LEVELS: dict[str, str] = {
    "RRSD": "L0B",
    "CRSD": "L0B",
    "RSLC": "L1",
    "RIFG": "L1",
    "RUNW": "L1",
    "ROFF": "L1",
    "GCOV": "L2",
    "GSLC": "L2",
    "GUNW": "L2",
    "GOFF": "L2",
    "SME2": "L3",
}

# L0B products were first released as PROVISIONAL; there are no BETA collections.
_NO_BETA = {"RRSD", "CRSD"}

# Products generated from an acquisition pair (reference/secondary dates in name).
PAIR_PRODUCTS = {"RIFG", "RUNW", "ROFF", "GUNW", "GOFF"}


def _normalize_maturity(maturity: str) -> str:
    value = maturity.lower()
    if value not in MATURITIES:
        raise ValueError(f"Unknown maturity {maturity!r}; expected one of {', '.join(MATURITIES)}")
    return value


def nisar_short_names(product_type: str, maturity: str = "provisional") -> list[str]:
    """Return ASF/CMR collection short names for a NISAR product and maturity.

    Args:
        product_type: NISAR product type (e.g. ``"GCOV"``, ``"GUNW"``).
        maturity: ``"provisional"``, ``"beta"``, ``"validated"``, or ``"any"``.

    Returns:
        List of collection short names. ``"any"`` returns every maturity that
        exists for the product.

    Raises:
        ValueError: If the product type or maturity is unknown, or if BETA is
            requested for a product with no BETA collection.
    """
    product = product_type.upper()
    if product not in PRODUCT_LEVELS:
        raise ValueError(
            f"Unknown NISAR product {product_type!r}; expected one of {', '.join(PRODUCT_LEVELS)}"
        )
    value = _normalize_maturity(maturity)
    level = PRODUCT_LEVELS[product]

    names = {
        "beta": f"NISAR_{level}_{product}_BETA_V1",
        "provisional": f"NISAR_{level}_{product}_PROVISIONAL_V1",
        "validated": f"NISAR_{level}_{product}_V1",
    }
    if product in _NO_BETA:
        if value == "beta":
            raise ValueError(f"No BETA collection exists for {product}")
        del names["beta"]

    if value == "any":
        return list(names.values())
    return [names[value]]


def maturity_from_collection(short_name: str | None) -> str | None:
    """Infer data maturity from a collection short name.

    Args:
        short_name: Collection short name, e.g. ``"NISAR_L2_GCOV_BETA_V1"``.

    Returns:
        ``"beta"``, ``"provisional"``, ``"validated"``, or ``None`` if the name
        is not a recognized NISAR collection.
    """
    if not short_name or not short_name.startswith("NISAR_"):
        return None
    if "_BETA_" in short_name:
        return "beta"
    if "_PROVISIONAL_" in short_name:
        return "provisional"
    if re.fullmatch(r"NISAR_L[0-3]B?_[A-Z0-9]{4}_V\d+", short_name):
        return "validated"
    return None


def maturity_from_crid(crid: str) -> str:
    """Infer data maturity from a Composite Release ID (CRID).

    CRIDs have the form ``EPMMmm`` (e.g. ``X05009``, ``P05023``). The release
    number (``PMMmm``) decides maturity: 05023 and higher is PROVISIONAL, lower is
    BETA. The leading environment letter is ignored because the PROVISIONAL
    collections also hold a few ``X05026`` granules (BETA collections contain
    only ``X05007``, ``X05009``, and ``X05010``; CMR census, September 2026).
    This cannot distinguish PROVISIONAL from VALIDATED; prefer the collection
    name (:func:`maturity_from_collection`) when it is available.

    Args:
        crid: Six-character CRID.

    Returns:
        ``"provisional"`` or ``"beta"``.
    """
    if not re.fullmatch(r"[A-Z]\d{5}", crid):
        raise ValueError(f"Invalid CRID {crid!r}; expected e.g. 'P05023'")
    if int(crid[1:]) >= int(PROVISIONAL_MIN_CRID[1:]):
        return "provisional"
    return "beta"


@dataclass(frozen=True)
class NisarGranuleName:
    """Fields parsed from a NISAR L1-L3 granule ID.

    For pair products (RIFG, RUNW, ROFF, GUNW, GOFF), ``start``/``end`` refer to
    the reference acquisition and ``secondary_start``/``secondary_end`` to the
    secondary acquisition.
    """

    granule_id: str
    level: str
    pipeline: str
    product: str
    cycle: int
    track: int
    direction: str
    frame: int
    mode: str
    polarization: str
    start: datetime
    end: datetime
    crid: str
    orbit_accuracy: str
    coverage: str
    counter: int
    secondary_cycle: int | None = None
    secondary_start: datetime | None = None
    secondary_end: datetime | None = None
    main_band: str | None = None

    @property
    def maturity(self) -> str:
        """Maturity implied by the CRID (see :func:`maturity_from_crid`)."""
        return maturity_from_crid(self.crid)

    @property
    def is_pair(self) -> bool:
        """Whether the product is formed from a reference/secondary pair."""
        return self.secondary_start is not None

    @property
    def full_frame(self) -> bool:
        """Whether the product covers the full frame (``F``) vs partial (``P``)."""
        return self.coverage == "F"


_TS = r"\d{8}T\d{6}"
_COMMON = r"NISAR_(?P<sensor>[LS])(?P<level>\d)_(?P<pipeline>PR|UR|OD)_(?P<product>[A-Z0-9]{4})"
_SINGLE_RE = re.compile(
    _COMMON + r"_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<direction>[AD])_(?P<frame>\d{3})"
    r"_(?P<mode>\d{4})_(?P<pol>[A-Z]{4})_(?P<main_band>[AM])"
    rf"_(?P<start>{_TS})_(?P<end>{_TS})_(?P<crid>[A-Z]\d{{5}})"
    r"_(?P<accuracy>[A-Z])_(?P<coverage>[FP])_(?P<loc>[A-Z])_(?P<counter>\d{3})$"
)
_PAIR_RE = re.compile(
    _COMMON + r"_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<direction>[AD])_(?P<frame>\d{3})"
    r"_(?P<secondary_cycle>\d{3})_(?P<mode>\d{4})_(?P<pol>[A-Z]{2})"
    rf"_(?P<start>{_TS})_(?P<end>{_TS})_(?P<sec_start>{_TS})_(?P<sec_end>{_TS})"
    r"_(?P<crid>[A-Z]\d{5})"
    r"_(?P<accuracy>[A-Z])_(?P<coverage>[FP])_(?P<loc>[A-Z])_(?P<counter>\d{3})$"
)


def _ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def parse_granule_name(name: str) -> NisarGranuleName:
    """Parse a NISAR L1-L3 granule ID or file name.

    Follows the NISAR file naming conventions for single-acquisition products
    (RSLC, GSLC, GCOV, SME2) and pair products (RIFG, RUNW, GUNW, ROFF, GOFF).

    Args:
        name: Granule ID or file name (directory and ``.h5`` suffix, as well as
            a trailing ``_QA_STATS``, are ignored).

    Returns:
        Parsed :class:`NisarGranuleName`.

    Raises:
        ValueError: If ``name`` does not match either naming layout.
    """
    stem = name.rsplit("/", 1)[-1]
    stem = re.sub(r"(_QA_STATS)?\.h5$", "", stem)

    match = _PAIR_RE.match(stem)
    if match:
        g = match.groupdict()
        return NisarGranuleName(
            granule_id=stem,
            level=f"L{g['level']}",
            pipeline=g["pipeline"],
            product=g["product"],
            cycle=int(g["cycle"]),
            track=int(g["track"]),
            direction=g["direction"],
            frame=int(g["frame"]),
            mode=g["mode"],
            polarization=g["pol"],
            start=_ts(g["start"]),
            end=_ts(g["end"]),
            crid=g["crid"],
            orbit_accuracy=g["accuracy"],
            coverage=g["coverage"],
            counter=int(g["counter"]),
            secondary_cycle=int(g["secondary_cycle"]),
            secondary_start=_ts(g["sec_start"]),
            secondary_end=_ts(g["sec_end"]),
        )

    match = _SINGLE_RE.match(stem)
    if match:
        g = match.groupdict()
        return NisarGranuleName(
            granule_id=stem,
            level=f"L{g['level']}",
            pipeline=g["pipeline"],
            product=g["product"],
            cycle=int(g["cycle"]),
            track=int(g["track"]),
            direction=g["direction"],
            frame=int(g["frame"]),
            mode=g["mode"],
            polarization=g["pol"],
            start=_ts(g["start"]),
            end=_ts(g["end"]),
            crid=g["crid"],
            orbit_accuracy=g["accuracy"],
            coverage=g["coverage"],
            counter=int(g["counter"]),
            main_band=g["main_band"],
        )

    raise ValueError(f"Not a recognized NISAR L1-L3 granule name: {name!r}")


#: Base URL of NISAR products in the ASF Earthdata Cloud archive.
ASF_NISAR_BASE_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR"


def granule_url(granule_id: str) -> str:
    """Return the ASF HTTPS URL of a NISAR granule from its ID.

    The URL follows the archive layout
    ``{base}/{collection}/{granule_id}/{granule_id}.h5``, with the collection
    chosen from the product type and the maturity implied by the CRID. This
    pins an exact granule without a search, which keeps tutorials and analyses
    reproducible. Opening the URL needs an Earthdata login (for example through
    :func:`nice_sar.auth.get_https_filesystem`).

    Args:
        granule_id: NISAR granule ID (a file name or ``.h5`` suffix is accepted).

    Returns:
        HTTPS URL of the granule's HDF5 file.
    """
    g = parse_granule_name(granule_id)
    collection = nisar_short_names(g.product, g.maturity)[0]
    return f"{ASF_NISAR_BASE_URL}/{collection}/{g.granule_id}/{g.granule_id}.h5"
