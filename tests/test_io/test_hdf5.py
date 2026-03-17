"""Integration tests for nice_sar.io.hdf5 using a synthetic GCOV HDF5 file."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from nice_sar.io.hdf5 import get_frequencies, get_polarizations, open_nisar


class TestOpenNisar:
    """Tests for open_nisar with local file paths."""

    def test_returns_h5py_file(self, synthetic_gcov_path: Path) -> None:
        h5 = open_nisar(synthetic_gcov_path)
        assert isinstance(h5, h5py.File)
        h5.close()

    def test_accepts_string_path(self, synthetic_gcov_path: Path) -> None:
        h5 = open_nisar(str(synthetic_gcov_path))
        assert isinstance(h5, h5py.File)
        h5.close()

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_nisar(tmp_path / "nonexistent.h5")

    def test_s3_path_without_filesystem(self) -> None:
        with pytest.raises(ValueError, match="filesystem"):
            open_nisar("s3://bucket/key.h5")


class TestGetFrequencies:
    """Tests for get_frequencies."""

    def test_returns_frequency_list(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            freqs = get_frequencies(h5)
        assert freqs == ["A"]

    def test_returns_list_of_strings(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            freqs = get_frequencies(h5)
        assert all(isinstance(f, str) for f in freqs)


class TestGetPolarizations:
    """Tests for get_polarizations."""

    def test_returns_polarization_list(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            pols = get_polarizations(h5, frequency="A")
        assert pols == ["HH", "HV"]

    def test_returns_strings(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5:
            pols = get_polarizations(h5, frequency="A")
        assert all(isinstance(p, str) for p in pols)

    def test_missing_frequency_raises(self, synthetic_gcov_path: Path) -> None:
        with h5py.File(synthetic_gcov_path, "r") as h5, pytest.raises(KeyError):
            get_polarizations(h5, frequency="B")
