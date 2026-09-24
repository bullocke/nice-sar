"""Live checks of maturity filtering against ASF/CMR.

Deselected by default. Run with ``pytest -m network``. No Earthdata login is
needed because only search metadata is queried.
"""

from __future__ import annotations

import pytest

from nice_sar.search.asf import search_nisar, summarize_results
from nice_sar.search.maturity import maturity_from_crid

pytestmark = pytest.mark.network

# Rondônia, Brazil: covered by both BETA (2025-26) and PROVISIONAL (2026-) data.
RONDONIA = (-63.5, -10.0, -62.5, -9.0)


@pytest.mark.parametrize("product", ["GCOV", "GUNW"])
def test_provisional_only(product: str) -> None:
    results = search_nisar(product, bbox=RONDONIA, max_results=20)
    assert results
    for s in summarize_results(results):
        assert s.maturity == "provisional"
        assert s.crid is not None and maturity_from_crid(s.crid) == "provisional"


@pytest.mark.parametrize("product", ["GCOV", "GUNW"])
def test_beta_only(product: str) -> None:
    results = search_nisar(product, bbox=RONDONIA, maturity="beta", max_results=20)
    assert results
    for s in summarize_results(results):
        assert s.maturity == "beta"
        assert s.crid is not None and s.crid < "X05023"


def test_any_returns_both() -> None:
    results = search_nisar("GCOV", bbox=RONDONIA, maturity="any", max_results=500)
    assert {s.maturity for s in summarize_results(results)} == {"beta", "provisional"}


def test_track_frame_filter() -> None:
    results = search_nisar("GCOV", track=161, frame=173, direction="A", max_results=20)
    assert results
    assert {(s.track, s.direction, s.frame) for s in summarize_results(results)} == {
        (161, "A", 173)
    }


def test_supplemental_frame_before_forward_processing() -> None:
    """Supplemental PROVISIONAL frames extend before 2026-06-17."""
    results = search_nisar("GCOV", track=25, frame=90, end="2026-06-16", max_results=5)
    assert results
    assert all(s.maturity == "provisional" for s in summarize_results(results))
