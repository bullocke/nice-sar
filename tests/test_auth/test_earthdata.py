"""Tests for nice_sar.auth.earthdata (mock-based, no real credentials)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLogin:
    """Tests for the login() function."""

    @patch("nice_sar.auth.earthdata.earthaccess")
    def test_login_success(self, mock_ea: MagicMock) -> None:
        from nice_sar.auth.earthdata import login

        mock_auth = MagicMock()
        mock_auth.authenticated = True
        mock_ea.login.return_value = mock_auth

        login()  # should not raise
        mock_ea.login.assert_called_once()

    @patch("nice_sar.auth.earthdata.earthaccess")
    def test_login_failure_raises(self, mock_ea: MagicMock) -> None:
        from nice_sar.auth.earthdata import login

        mock_auth = MagicMock()
        mock_auth.authenticated = False
        mock_ea.login.return_value = mock_auth

        with pytest.raises(RuntimeError, match="authentication failed"):
            login()


class TestGetS3Filesystem:
    """Tests for get_s3_filesystem()."""

    @patch("nice_sar.auth.earthdata.s3fs")
    @patch("nice_sar.auth.earthdata.earthaccess")
    def test_returns_s3_filesystem(
        self, mock_ea: MagicMock, mock_s3fs: MagicMock
    ) -> None:
        from nice_sar.auth.earthdata import get_s3_filesystem

        mock_auth = MagicMock()
        mock_auth.get_s3_credentials.return_value = {
            "accessKeyId": "AKID",
            "secretAccessKey": "SECRET",
            "sessionToken": "TOKEN",
        }
        mock_ea.login.return_value = mock_auth

        mock_fs = MagicMock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        result = get_s3_filesystem()

        assert result is mock_fs
        mock_s3fs.S3FileSystem.assert_called_once_with(
            key="AKID", secret="SECRET", token="TOKEN"
        )

    @patch("nice_sar.auth.earthdata.s3fs")
    @patch("nice_sar.auth.earthdata.earthaccess")
    def test_uses_nisar_endpoint(
        self, mock_ea: MagicMock, mock_s3fs: MagicMock
    ) -> None:
        from nice_sar.auth.earthdata import _NISAR_S3_ENDPOINT, get_s3_filesystem

        mock_auth = MagicMock()
        mock_auth.get_s3_credentials.return_value = {
            "accessKeyId": "A",
            "secretAccessKey": "B",
            "sessionToken": "C",
        }
        mock_ea.login.return_value = mock_auth
        mock_s3fs.S3FileSystem.return_value = MagicMock()

        get_s3_filesystem()
        mock_auth.get_s3_credentials.assert_called_once_with(
            endpoint=_NISAR_S3_ENDPOINT
        )


class TestGetHttpsFilesystem:
    """Tests for get_https_filesystem()."""

    @patch("nice_sar.auth.earthdata.earthaccess")
    def test_returns_filesystem(self, mock_ea: MagicMock) -> None:
        from nice_sar.auth.earthdata import get_https_filesystem

        mock_fs = MagicMock()
        mock_ea.get_fsspec_https_session.return_value = mock_fs

        result = get_https_filesystem()
        assert result is mock_fs
        mock_ea.get_fsspec_https_session.assert_called_once()
