"""Tests for nice_sar.search.maturity (no network)."""

from __future__ import annotations

from datetime import datetime

import pytest

from nice_sar.search.maturity import (
    PRODUCT_LEVELS,
    maturity_from_collection,
    maturity_from_crid,
    nisar_short_names,
    parse_granule_name,
)

GCOV_PROV = (
    "NISAR_L2_PR_GCOV_004_004_A_018_4005_DHDH_A_20251029T111130_20251029T111153_P05023_N_P_J_001"
)
GCOV_BETA = (
    "NISAR_L2_PR_GCOV_005_172_A_008_2005_DHDH_A_20251122T024618_20251122T024652_X05007_N_F_J_001"
)
GUNW_PROV = (
    "NISAR_L2_PR_GUNW_004_004_A_018_006_4000_SH_20251029T111130_20251029T111153"
    "_20251122T111119_20251122T111153_P05023_N_P_J_001"
)


class TestShortNames:
    @pytest.mark.parametrize(
        ("product", "maturity", "expected"),
        [
            ("GCOV", "provisional", ["NISAR_L2_GCOV_PROVISIONAL_V1"]),
            ("gcov", "BETA", ["NISAR_L2_GCOV_BETA_V1"]),
            ("GUNW", "validated", ["NISAR_L2_GUNW_V1"]),
            ("RSLC", "provisional", ["NISAR_L1_RSLC_PROVISIONAL_V1"]),
            ("SME2", "beta", ["NISAR_L3_SME2_BETA_V1"]),
            ("RRSD", "provisional", ["NISAR_L0B_RRSD_PROVISIONAL_V1"]),
        ],
    )
    def test_mapping(self, product: str, maturity: str, expected: list[str]) -> None:
        assert nisar_short_names(product, maturity) == expected

    def test_any_returns_all(self) -> None:
        assert nisar_short_names("GSLC", "any") == [
            "NISAR_L2_GSLC_BETA_V1",
            "NISAR_L2_GSLC_PROVISIONAL_V1",
            "NISAR_L2_GSLC_V1",
        ]

    def test_any_without_beta_for_l0b(self) -> None:
        assert nisar_short_names("CRSD", "any") == [
            "NISAR_L0B_CRSD_PROVISIONAL_V1",
            "NISAR_L0B_CRSD_V1",
        ]

    def test_default_is_provisional(self) -> None:
        assert nisar_short_names("GCOV") == ["NISAR_L2_GCOV_PROVISIONAL_V1"]

    def test_every_product_resolves(self) -> None:
        for product in PRODUCT_LEVELS:
            assert nisar_short_names(product, "provisional")

    def test_unknown_product(self) -> None:
        with pytest.raises(ValueError, match="Unknown NISAR product"):
            nisar_short_names("GRD")

    def test_unknown_maturity(self) -> None:
        with pytest.raises(ValueError, match="Unknown maturity"):
            nisar_short_names("GCOV", "stable")

    def test_beta_l0b_raises(self) -> None:
        with pytest.raises(ValueError, match="No BETA"):
            nisar_short_names("RRSD", "beta")


class TestMaturityInference:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("NISAR_L2_GCOV_BETA_V1", "beta"),
            ("NISAR_L2_GUNW_PROVISIONAL_V1", "provisional"),
            ("NISAR_L2_GCOV_V1", "validated"),
            ("NISAR_L0B_RRSD_V1", "validated"),
            ("SENTINEL-1A_SLC", None),
            (None, None),
        ],
    )
    def test_from_collection(self, name: str | None, expected: str | None) -> None:
        assert maturity_from_collection(name) == expected

    @pytest.mark.parametrize(
        ("crid", "expected"),
        [
            ("X05007", "beta"),
            ("X05009", "beta"),
            ("X05010", "beta"),
            ("P05023", "provisional"),
            ("X05026", "provisional"),
            ("P05100", "provisional"),
        ],
    )
    def test_from_crid(self, crid: str, expected: str) -> None:
        assert maturity_from_crid(crid) == expected

    def test_invalid_crid(self) -> None:
        with pytest.raises(ValueError, match="Invalid CRID"):
            maturity_from_crid("05023")


class TestParseGranuleName:
    def test_single_acquisition(self) -> None:
        g = parse_granule_name(GCOV_PROV)
        assert g.product == "GCOV"
        assert g.level == "L2"
        assert g.pipeline == "PR"
        assert (g.cycle, g.track, g.direction, g.frame) == (4, 4, "A", 18)
        assert g.mode == "4005"
        assert g.polarization == "DHDH"
        assert g.main_band == "A"
        assert g.start == datetime(2025, 10, 29, 11, 11, 30)
        assert g.crid == "P05023"
        assert g.maturity == "provisional"
        assert not g.full_frame
        assert not g.is_pair
        assert g.counter == 1

    def test_beta_name(self) -> None:
        g = parse_granule_name(GCOV_BETA)
        assert g.maturity == "beta"
        assert (g.track, g.frame) == (172, 8)
        assert g.full_frame

    def test_pair_product(self) -> None:
        g = parse_granule_name(GUNW_PROV)
        assert g.product == "GUNW"
        assert g.is_pair
        assert g.secondary_cycle == 6
        assert g.polarization == "SH"
        assert g.start == datetime(2025, 10, 29, 11, 11, 30)
        assert g.secondary_start == datetime(2025, 11, 22, 11, 11, 19)
        assert g.main_band is None

    @pytest.mark.parametrize("suffix", [".h5", "_QA_STATS.h5"])
    def test_file_names_and_paths(self, suffix: str) -> None:
        g = parse_granule_name(f"/data/nisar/{GCOV_PROV}{suffix}")
        assert g.granule_id == GCOV_PROV

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Not a recognized"):
            parse_granule_name("S1A_IW_SLC__1SDV_20250101T000000")


class TestGranuleUrl:
    def test_provisional_gcov(self) -> None:
        from nice_sar.search.maturity import granule_url

        gid = (
            "NISAR_L2_PR_GCOV_023_068_D_093_2005_DHDH_A_20260618T222633_20260618T222707"
            "_P05023_N_F_J_001"
        )
        assert granule_url(gid + ".h5") == (
            "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1/"
            f"{gid}/{gid}.h5"
        )

    def test_gunw_pair(self) -> None:
        from nice_sar.search.maturity import granule_url

        gid = (
            "NISAR_L2_PR_GUNW_004_083_D_088_006_2000_SH_20251103T232159_20251103T232233"
            "_20251127T232159_20251127T232233_P05023_N_F_J_001"
        )
        assert "/NISAR_L2_GUNW_PROVISIONAL_V1/" in granule_url(gid)
