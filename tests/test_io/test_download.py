"""Tests for nice_sar.io.download (mock-based, no real downloads)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDownloadUrl:
    """Tests for download_url()."""

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_creates_output_dir(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_url

        out = tmp_path / "sub" / "dir"
        mock_session.return_value = MagicMock()
        download_url("https://example.com/NISAR_GCOV.h5", out)
        assert out.exists()

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_returns_expected_path(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_url

        mock_session.return_value = MagicMock()
        result = download_url("https://example.com/NISAR_GCOV.h5", tmp_path)
        assert result == tmp_path / "NISAR_GCOV.h5"

    @patch("nice_sar.io.download.asf_search")
    def test_uses_provided_session(self, mock_asf: MagicMock, tmp_path: Path) -> None:
        from nice_sar.io.download import download_url

        session = MagicMock()
        download_url("https://example.com/file.h5", tmp_path, session=session)
        mock_asf.download_url.assert_called_once_with(
            url="https://example.com/file.h5",
            path=str(tmp_path),
            session=session,
        )


class TestDownloadGranules:
    """Tests for download_granules()."""

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_creates_output_dir(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_granules

        out = tmp_path / "downloads"
        mock_session.return_value = MagicMock()
        download_granules([], out)
        assert out.exists()

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_calls_download_urls(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_granules

        mock_session.return_value = MagicMock()
        r1 = MagicMock()
        r1.properties = {"url": "https://example.com/g1.h5"}
        r2 = MagicMock()
        r2.properties = {"url": "https://example.com/g2.h5"}

        download_granules([r1, r2], tmp_path)
        mock_asf.download_urls.assert_called_once()
        call_kwargs = mock_asf.download_urls.call_args
        assert len(call_kwargs[1]["urls"]) == 2

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_returns_paths_for_results_only(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_granules

        (tmp_path / "unrelated.h5").write_bytes(b"x")
        r1 = MagicMock()
        r1.properties = {"url": "https://example.com/g1.h5", "fileName": "g1.h5"}
        paths = download_granules([r1], tmp_path)
        assert paths == [tmp_path / "g1.h5"]

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_skips_complete_and_replaces_partial(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_granules

        (tmp_path / "done.h5").write_bytes(b"1234")
        (tmp_path / "partial.h5").write_bytes(b"12")
        done = MagicMock()
        done.properties = {
            "url": "https://example.com/done.h5",
            "fileName": "done.h5",
            "bytes": {"done.h5": {"bytes": 4}},
        }
        partial = MagicMock()
        partial.properties = {
            "url": "https://example.com/partial.h5",
            "fileName": "partial.h5",
            "bytes": {"partial.h5": {"bytes": 4}},
        }
        download_granules([done, partial], tmp_path)
        urls = mock_asf.download_urls.call_args.kwargs["urls"]
        assert urls == ["https://example.com/partial.h5"]
        assert not (tmp_path / "partial.h5").exists()

    @patch("nice_sar.io.download._get_asf_session")
    @patch("nice_sar.io.download.asf_search")
    def test_no_session_when_all_present(
        self, mock_asf: MagicMock, mock_session: MagicMock, tmp_path: Path
    ) -> None:
        from nice_sar.io.download import download_granules

        (tmp_path / "done.h5").write_bytes(b"1234")
        done = MagicMock()
        done.properties = {
            "url": "https://example.com/done.h5",
            "fileName": "done.h5",
            "bytes": {"done.h5": {"bytes": 4}},
        }
        assert download_granules([done], tmp_path) == [tmp_path / "done.h5"]
        mock_session.assert_not_called()
        mock_asf.download_urls.assert_not_called()


class TestGetAsfSession:
    """Tests for _get_asf_session()."""

    @patch("nice_sar.io.download.asf_search")
    @patch("nice_sar.io.download.earthaccess")
    def test_creates_session_with_token(self, mock_ea: MagicMock, mock_asf: MagicMock) -> None:
        from nice_sar.io.download import _get_asf_session

        mock_auth = MagicMock()
        mock_auth.authenticated = True
        mock_auth.token = {"access_token": "test-token-123"}
        mock_ea.login.return_value = mock_auth

        mock_session = MagicMock()
        mock_asf.ASFSession.return_value = mock_session

        result = _get_asf_session()
        assert result is mock_session
        mock_session.headers.update.assert_called_once_with(
            {"Authorization": "Bearer test-token-123"}
        )

    @patch("nice_sar.io.download.asf_search")
    @patch("nice_sar.io.download.earthaccess")
    def test_raises_on_unauthenticated(self, mock_ea: MagicMock, mock_asf: MagicMock) -> None:
        from nice_sar.io.download import _get_asf_session

        mock_auth = MagicMock()
        mock_auth.authenticated = False
        mock_ea.login.return_value = mock_auth

        with pytest.raises(RuntimeError, match="authentication required"):
            _get_asf_session()
