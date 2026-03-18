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
    def test_uses_provided_session(
        self, mock_asf: MagicMock, tmp_path: Path
    ) -> None:
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


class TestGetAsfSession:
    """Tests for _get_asf_session()."""

    @patch("nice_sar.io.download.asf_search")
    @patch("nice_sar.io.download.earthaccess")
    def test_creates_session_with_token(
        self, mock_ea: MagicMock, mock_asf: MagicMock
    ) -> None:
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
    def test_raises_on_unauthenticated(
        self, mock_ea: MagicMock, mock_asf: MagicMock
    ) -> None:
        from nice_sar.io.download import _get_asf_session

        mock_auth = MagicMock()
        mock_auth.authenticated = False
        mock_ea.login.return_value = mock_auth

        with pytest.raises(RuntimeError, match="authentication required"):
            _get_asf_session()
