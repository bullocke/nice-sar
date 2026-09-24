"""Shared paths, study-area constants, and method thresholds for the Caquetá analysis.

Every script in ``scripts/caqueta/`` imports from here, so a threshold changed in
this file changes it everywhere. Values chosen in earlier exploration are noted
with their rationale.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- Inputs (gitignored under NISAR_Data/) ----------------------------------------
DATA_DIR = REPO / "NISAR_Data" / "caqueta"
MANIFEST = DATA_DIR / "manifest.json"  # written by fetch_nisar_subsets.py
RADD_TIF = DATA_DIR / "RADD_reference.tif"  # written by fetch_radd.py
S2_DIR = DATA_DIR / "S2"  # written by fetch_sentinel2.py

# --- Outputs (gitignored) ---------------------------------------------------------
OUT_DIR = REPO / "local_examples" / "caqueta"

# --- Study area --------------------------------------------------------------------
# Densest cluster of RADD alerts inside NISAR frame 083 D 088 (supplemental
# PROVISIONAL frame, Nov 2025 onward). west, south, east, north (WGS84).
AOI = (-74.36, 0.76, -74.16, 0.92)
TRACK, DIRECTION, FRAME = 83, "D", 88
PIXEL_M = 20.0  # NISAR GCOV / GUNW-20 m grid spacing (EPSG:32618)

# --- Time --------------------------------------------------------------------------
# All day numbers in the analysis are days since EPOCH (RADD_reference.tif uses the
# same convention).
EPOCH = date(2025, 1, 1)


def to_date(day: float) -> date:
    """Convert a day number (days since EPOCH) to a calendar date."""
    return EPOCH + timedelta(days=int(round(day)))


def to_day(d: date | str) -> int:
    """Convert a date or ISO string to a day number (days since EPOCH)."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    return (d - EPOCH).days


# --- Pixel classes (from RADD) ----------------------------------------------------
STABLE_BUFFER_M = 100.0  # stable forest must be at least this far from any alert
PRE_SERIES_MARGIN_D = 90  # "cleared before series": alert >= 90 d before first date

# --- HV event dating ---------------------------------------------------------------
HV_BOXCAR_PX = 3  # 3x3 boxcar (linear power) before converting to dB
HV_MIN_DATES_EACH_SIDE = 2
HV_STEP_MIN_DB = 2.0  # clearing = HV falls >= 2 dB (forest ~-10 dB, pasture ~-12.5)
HV_NO_STEP_DB = 1.0  # "no NISAR response" below this
HV_RADD_MAX_OFFSET_D = 90  # bracket must end within 90 d of the RADD alert date
CORE_EROSION_PX = 2  # drop pixels within 40 m of a clearing edge

# --- Coherence test ----------------------------------------------------------------
PRE_PAIR_BUFFER_D = 12  # pre-event pairs must end >= 12 d before the bracket
MIN_PRE_PAIRS = 4  # pixel needs >= 4 pre-event pairs to define its own baseline
SEED = 20260924

# GUNW multilooking, from the product metadata
# /science/LSAR/GUNW/metadata/processingInformation/parameters/
#   wrappedInterferogram/frequencyA/numberOf{Azimuth,Range}Looks = 6, 3  (20 m)
#   unwrappedInterferogram/frequencyA/numberOf{Azimuth,Range}Looks = 16, 7 (80 m)
LOOKS = {"coh20": 6 * 3, "coh80": 16 * 7}

# --- Sentinel-2 chips --------------------------------------------------------------
S2_MIN_CLEAR = 0.8  # chip must be >= 80% clear (Cloud Score+ cs_cdf >= 0.6)
# Cloud Score+ passes thin haze (often smoke in the Feb-Mar burning season). Clear
# chips here have median blue (B2) reflectance of 0.023-0.053; hazy ones 0.065-0.13.
S2_MAX_BLUE = 600  # median B2 of clear pixels, reflectance x 10000

# --- Case studies ------------------------------------------------------------------
MIN_PATCH_PX = 25  # 25 px x 400 m2 = 1 ha
# Patch-level rules (forest-normalized patch-mean HV). Patch steps are smaller than
# pixel steps because a RADD patch mixes fully and partly cleared pixels: across
# 417 candidate patches the median step is 1.4 dB and only 11% reach 2 dB.
PATCH_STEP_DB = 1.5
PATCH_NO_STEP_DB = 0.75
ABRUPT_FRAC = 0.6  # >= 60% of the step happens between two consecutive HV dates
GRADUAL_FRAC = 0.4  # < 40%: decline spread over several dates
DIP_Z = -1.5  # lowest in-bracket 80 m coherence >= 1.5 pre-event SDs below baseline
CASES_PER_CATEGORY = 3
