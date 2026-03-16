"""Shared test fixtures for nice-sar.

Provides synthetic NISAR-like data for unit tests without
requiring real satellite data.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.create_synthetic_gcov import create_synthetic_gcov


@pytest.fixture(scope="session")
def synthetic_gcov_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a synthetic NISAR GCOV HDF5 file, shared across all tests."""
    path = tmp_path_factory.mktemp("fixtures") / "synthetic_gcov.h5"
    return create_synthetic_gcov(path)


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def hh_linear(rng: np.random.Generator) -> np.ndarray:
    """Synthetic HH backscatter in linear power (100x100)."""
    return rng.exponential(0.05, size=(100, 100)).astype(np.float32)


@pytest.fixture()
def hv_linear(rng: np.random.Generator) -> np.ndarray:
    """Synthetic HV backscatter in linear power (100x100)."""
    return rng.exponential(0.01, size=(100, 100)).astype(np.float32)


@pytest.fixture()
def hh_db(hh_linear: np.ndarray) -> np.ndarray:
    """Synthetic HH backscatter in dB."""
    return (10 * np.log10(hh_linear + 1e-10)).astype(np.float32)


@pytest.fixture()
def hv_db(hv_linear: np.ndarray) -> np.ndarray:
    """Synthetic HV backscatter in dB."""
    return (10 * np.log10(hv_linear + 1e-10)).astype(np.float32)


@pytest.fixture()
def vv_linear(rng: np.random.Generator) -> np.ndarray:
    """Synthetic VV backscatter in linear power (100x100)."""
    return rng.exponential(0.04, size=(100, 100)).astype(np.float32)
