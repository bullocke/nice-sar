"""Tests for nice_sar.search — mock-based (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nice_sar.search.asf import get_result_size_bytes, search_gcov, search_nisar
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
        assert kwargs["dataset"] == "NISAR"
        assert kwargs["processingLevel"] == "GCOV"
        assert kwargs["maxResults"] == 100

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
        assert kwargs["processingLevel"] == "RSLC"

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
        assert kwargs["processingLevel"] == "GCOV"
        assert kwargs["maxResults"] == 5


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
        assert kwargs["short_name"] == "NISAR_L2_GCOV_BETA_V1"
        assert kwargs["count"] == 100

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
