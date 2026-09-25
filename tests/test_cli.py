"""Tests for nice_sar.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nice_sar.cli import build_parser, cmd_forests, cmd_info, main


class TestBuildParser:
    def test_parser_builds(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_info_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", "/tmp/test.h5"])
        assert args.command == "info"

    def test_read_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["read", "/tmp/in.h5", "/tmp/out.tif", "-p", "GCOV"]
        )
        assert args.command == "read"
        assert args.product == "GCOV"

    def test_multilook_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["multilook", "/tmp/in.h5", "/tmp/out.tif", "--looks-y", "4"]
        )
        assert args.command == "multilook"
        assert args.looks_y == 4

    def test_insar_phase2disp(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["insar", "phase2disp", "/tmp/in.h5", "/tmp/out.tif", "--apply-iono"]
        )
        assert args.command == "insar"
        assert args.subcommand == "phase2disp"
        assert args.apply_iono is True

    def test_timeseries_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["timeseries", "f1.h5", "f2.h5", "-o", "out.tif", "--method", "cusum"]
        )
        assert args.command == "timeseries"
        assert args.method == "cusum"
        assert len(args.inputs) == 2

    def test_forests_generate_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "forests",
                "generate",
                "/tmp/in.h5",
                "/tmp/out.tif",
                "--method",
                "gcov_hv_threshold",
            ]
        )
        assert args.command == "forests"
        assert args.subcommand == "generate"
        assert args.method == "gcov_hv_threshold"

    def test_forests_list_methods_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["forests", "list-methods", "--implemented-only"])
        assert args.command == "forests"
        assert args.subcommand == "list-methods"
        assert args.implemented_only is True


class TestCmdInfo:
    def test_info_prints_json(self, synthetic_gcov_path: Path, capsys: pytest.CaptureFixture) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", str(synthetic_gcov_path)])
        cmd_info(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["product_type"] == "GCOV"
        assert "frequencies" in data
        assert "polarizations" in data

    def test_info_gunw(self, synthetic_gunw_path: Path, capsys: pytest.CaptureFixture) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", str(synthetic_gunw_path)])
        cmd_info(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["product_type"] == "GUNW"


class TestCmdForests:
    def test_list_methods_prints_json(self, capsys: pytest.CaptureFixture) -> None:
        parser = build_parser()
        args = parser.parse_args(["forests", "list-methods"])
        cmd_forests(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        names = {item["name"] for item in data}
        assert "gcov_hv_threshold" in names

    def test_generate_writes_mask_and_confidence(
        self,
        synthetic_gcov_path: Path,
        tmp_path: Path,
    ) -> None:
        parser = build_parser()
        output = tmp_path / "forest_mask.tif"
        confidence = tmp_path / "forest_confidence.tif"
        args = parser.parse_args(
            [
                "forests",
                "generate",
                str(synthetic_gcov_path),
                str(output),
                "--method",
                "gcov_hv_threshold_ramachandran",
                "--confidence-output",
                str(confidence),
            ]
        )
        cmd_forests(args)
        assert output.exists()
        assert confidence.exists()


class TestSearchDownloadCommands:
    def test_search_defaults(self) -> None:
        args = build_parser().parse_args(["search"])
        assert args.command == "search"
        assert args.product == "GCOV"
        assert args.maturity == "provisional"
        assert args.bbox is None

    def test_search_filters(self) -> None:
        args = build_parser().parse_args(
            [
                "search",
                "--product",
                "GUNW",
                "--bbox=-63.5,-10,-62.5,-9",
                "--maturity",
                "any",
                "--track",
                "161",
                "--frame",
                "173",
                "--direction",
                "a",
                "--json",
            ]
        )
        assert (args.maturity, args.track, args.frame, args.direction) == (
            "any",
            161,
            173,
            "A",
        )
        assert args.json

    def test_invalid_maturity_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["search", "--maturity", "gamma"])

    def test_download_and_subset_accept_maturity(self) -> None:
        args = build_parser().parse_args(
            ["download", "--product", "GCOV", "--maturity", "beta", "-o", "out"]
        )
        assert args.maturity == "beta"
        assert args.output_dir == Path("out")
        args = build_parser().parse_args(
            ["subset", "--bbox=0,0,1,1", "--product", "GCOV", "--maturity", "any"]
        )
        assert args.maturity == "any"

    def test_cmd_search_passes_filters(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from unittest.mock import MagicMock

        calls: dict = {}

        def fake_search(**kwargs: object) -> list:
            calls.update(kwargs)
            r = MagicMock()
            r.properties = {
                "fileID": "NISAR_L2_PR_GCOV_023_060_A_172_2005_DHDH_A_20260618T095228"
                "_20260618T095302_P05023_N_F_J_001",
                "collectionName": "NISAR_L2_GCOV_PROVISIONAL_V1",
            }
            return [r]

        monkeypatch.setattr("nice_sar.search.asf.search_nisar", fake_search)
        main(["search", "--maturity", "beta", "--track", "60", "--json"])
        assert calls["maturity"] == "beta"
        assert calls["track"] == 60
        out = json.loads(capsys.readouterr().out)
        assert out[0]["maturity"] == "provisional"
        assert out[0]["frame"] == 172
