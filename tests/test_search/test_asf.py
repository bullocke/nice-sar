"""Tests for nice_sar.search — mock-based (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nice_sar.search.asf import (
    get_result_size_bytes,
    search_gcov,
    search_gunw,
    search_nisar,
    summarize_results,
)
from nice_sar.search.earthdata import search_earthdata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_asf_result(name: str = "NISAR_GCOV_001.h5") -> MagicMock:
    """Create a minimal mock ASF search result."""
    r = MagicMock()
    r.properties = {"fileName": name, "processingLevel": "GCOV"}
    r.__str__ = lambda self: name
    return r


# ---------------------------------------------------------------------------
# search_nisar / search_gcov
# ---------------------------------------------------------------------------


class TestSearchNisar:
    """Tests for search_nisar()."""

    @patch("nice_sar.search.asf.asf_search.search")
    def test_returns_list(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [_fake_asf_result(), _fake_asf_result()]
        results = search_nisar()
        assert isinstance(results, list)
        assert len(results) == 2

    @patch("nice_sar.search.asf.asf_search.search")
    def test_default_kwargs(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar()
        mock_search.assert_called_once()
        kwargs = mock_search.call_args.kwargs
        assert kwargs["shortName"] == ["NISAR_L2_GCOV_PROVISIONAL_V1"]
        assert kwargs["maxResults"] == 100
        assert "cmr_keywords" not in kwargs
        assert "flightDirection" not in kwargs

    @patch("nice_sar.search.asf.asf_search.search")
    def test_beta_maturity(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(maturity="beta")
        assert mock_search.call_args.kwargs["shortName"] == ["NISAR_L2_GCOV_BETA_V1"]

    @patch("nice_sar.search.asf.asf_search.search")
    def test_any_maturity(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(product_type="GUNW", maturity="any")
        assert mock_search.call_args.kwargs["shortName"] == [
            "NISAR_L2_GUNW_BETA_V1",
            "NISAR_L2_GUNW_PROVISIONAL_V1",
            "NISAR_L2_GUNW_V1",
        ]

    def test_invalid_maturity_raises(self) -> None:
        with pytest.raises(ValueError, match="maturity"):
            search_nisar(maturity="gamma")

    @patch("nice_sar.search.asf.asf_search.search")
    def test_track_frame_direction(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(track=161, frame=173, direction="a")
        kwargs = mock_search.call_args.kwargs
        assert kwargs["flightDirection"] == "ASCENDING"
        assert kwargs["cmr_keywords"] == [
            ("attribute[]", "int,TRACK_NUMBER,161"),
            ("attribute[]", "int,FRAME_NUMBER,173"),
        ]

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            search_nisar(direction="north")

    @patch("nice_sar.search.asf.asf_search.search")
    def test_bbox_converted_to_wkt(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(bbox=(-112.0, 40.0, -111.0, 41.0))
        kwargs = mock_search.call_args.kwargs
        assert "intersectsWith" in kwargs
        assert "POLYGON" in kwargs["intersectsWith"]

    @patch("nice_sar.search.asf.asf_search.search")
    def test_start_end_passed(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(start="2025-01-01", end="2025-12-31")
        kwargs = mock_search.call_args.kwargs
        assert kwargs["start"] == "2025-01-01"
        assert kwargs["end"] == "2025-12-31"

    @patch("nice_sar.search.asf.asf_search.search")
    def test_product_type_override(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(product_type="RSLC")
        kwargs = mock_search.call_args.kwargs
        assert kwargs["shortName"] == ["NISAR_L1_RSLC_PROVISIONAL_V1"]

    @patch("nice_sar.search.asf.asf_search.search")
    def test_max_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_nisar(max_results=10)
        kwargs = mock_search.call_args.kwargs
        assert kwargs["maxResults"] == 10


class TestSearchGcov:
    """Tests for search_gcov() convenience wrapper."""

    @patch("nice_sar.search.asf.asf_search.search")
    def test_delegates_to_search_nisar(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [_fake_asf_result()]
        results = search_gcov(bbox=(-112.0, 40.0, -111.0, 41.0), max_results=5)
        assert len(results) == 1
        kwargs = mock_search.call_args.kwargs
        assert kwargs["shortName"] == ["NISAR_L2_GCOV_PROVISIONAL_V1"]
        assert kwargs["maxResults"] == 5

    @patch("nice_sar.search.asf.asf_search.search")
    def test_passes_maturity_and_filters(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_gcov(maturity="beta", track=25)
        kwargs = mock_search.call_args.kwargs
        assert kwargs["shortName"] == ["NISAR_L2_GCOV_BETA_V1"]
        assert kwargs["cmr_keywords"] == [("attribute[]", "int,TRACK_NUMBER,25")]


class TestSearchGunw:
    """Tests for search_gunw() convenience wrapper."""

    @patch("nice_sar.search.asf.asf_search.search")
    def test_uses_gunw_collection(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_gunw()
        assert mock_search.call_args.kwargs["shortName"] == ["NISAR_L2_GUNW_PROVISIONAL_V1"]


GCOV_ID = (
    "NISAR_L2_PR_GCOV_023_060_A_172_2005_DHDH_A_20260618T095228_20260618T095302_P05023_N_F_J_001"
)
GUNW_ID = (
    "NISAR_L2_PR_GUNW_029_161_A_174_030_2000_SH_20260905T100151_20260905T100225"
    "_20260917T100151_20260917T100225_P05023_N_F_J_001"
)


class TestSummarizeResults:
    """Tests for summarize_results()."""

    def test_uses_collection_and_properties(self) -> None:
        r = MagicMock()
        r.properties = {
            "fileID": GUNW_ID,
            "fileName": GUNW_ID + ".h5",
            "collectionName": "NISAR_L2_GUNW_PROVISIONAL_V1",
            "crid": "P05023",
            "processingLevel": "GUNW",
            "pathNumber": 161,
            "frameNumber": 174,
            "bytes": {GUNW_ID + ".h5": {"bytes": 2_280_000_000}},
            "url": "https://example.com/g.h5",
        }
        (s,) = summarize_results([r])
        assert s.maturity == "provisional"
        assert s.crid == "P05023"
        assert (s.track, s.direction, s.frame) == (161, "A", 174)
        assert s.start == "2026-09-05T10:01:51"
        assert s.secondary_start == "2026-09-17T10:01:51"
        assert s.size_gb == 2.28
        assert s.to_dict()["url"] == "https://example.com/g.h5"

    def test_falls_back_to_granule_name(self) -> None:
        r = MagicMock()
        r.properties = {"sceneName": GCOV_ID.replace("P05023", "X05009")}
        (s,) = summarize_results([r])
        assert s.maturity == "beta"
        assert s.crid == "X05009"
        assert (s.product, s.track, s.frame) == ("GCOV", 60, 172)
        assert s.full_frame is True
        assert s.secondary_start is None


class TestGetResultSizeBytes:
    """Tests for extracting ASF result file sizes across metadata schemas."""

    def test_returns_scalar_size(self) -> None:
        result = {"bytes": 123456789}
        assert get_result_size_bytes(result) == 123456789

    def test_returns_nisar_size_for_matching_filename(self) -> None:
        result = {
            "fileName": "granule.h5",
            "bytes": {"granule.h5": {"bytes": 987654321, "format": "HDF5"}},
        }
        assert get_result_size_bytes(result) == 987654321

    def test_returns_nested_scalar_size(self) -> None:
        result = {"bytes": {"bytes": 42}}
        assert get_result_size_bytes(result) == 42

    def test_returns_single_entry_size_without_filename(self) -> None:
        result = {"bytes": {"granule.h5": {"bytes": 314159265}}}
        assert get_result_size_bytes(result) == 314159265

    def test_returns_none_for_unrecognized_shape(self) -> None:
        result = {"bytes": {"granule.h5": {"format": "HDF5"}}}
        assert get_result_size_bytes(result) is None

    def test_accepts_result_object_with_properties(self) -> None:
        result = MagicMock()
        result.properties = {"fileName": "granule.h5", "bytes": {"granule.h5": {"bytes": 7}}}
        assert get_result_size_bytes(result) == 7


# ---------------------------------------------------------------------------
# search_earthdata
# ---------------------------------------------------------------------------


class TestSearchEarthdata:
    """Tests for search_earthdata()."""

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_returns_list(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [MagicMock(), MagicMock()]
        results = search_earthdata()
        assert isinstance(results, list)
        assert len(results) == 2

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_default_short_name(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata()
        kwargs = mock_search.call_args.kwargs
        assert kwargs["short_name"] == "NISAR_L2_GCOV_PROVISIONAL_V1"
        assert kwargs["count"] == 100

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_product_and_maturity(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata(product_type="GUNW", maturity="beta")
        assert mock_search.call_args.kwargs["short_name"] == "NISAR_L2_GUNW_BETA_V1"

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_any_maturity_passes_list(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata(maturity="any")
        assert mock_search.call_args.kwargs["short_name"] == [
            "NISAR_L2_GCOV_BETA_V1",
            "NISAR_L2_GCOV_PROVISIONAL_V1",
            "NISAR_L2_GCOV_V1",
        ]

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_custom_short_name(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata(short_name="NISAR_L2_GSLC_BETA_V1")
        kwargs = mock_search.call_args.kwargs
        assert kwargs["short_name"] == "NISAR_L2_GSLC_BETA_V1"

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_bbox_passed(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata(bbox=(-112.0, 40.0, -111.0, 41.0))
        kwargs = mock_search.call_args.kwargs
        assert kwargs["bounding_box"] == (-112.0, 40.0, -111.0, 41.0)

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_temporal_passed(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata(temporal=("2025-01-01", "2025-12-31"))
        kwargs = mock_search.call_args.kwargs
        assert kwargs["temporal"] == ("2025-01-01", "2025-12-31")

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_no_bbox_omits_key(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata()
        kwargs = mock_search.call_args.kwargs
        assert "bounding_box" not in kwargs

    @patch("nice_sar.search.earthdata.earthaccess.search_data")
    def test_no_temporal_omits_key(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        search_earthdata()
        kwargs = mock_search.call_args.kwargs
        assert "temporal" not in kwargs
