"""Tests for nice_sar.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nice_sar.cli import build_parser, cmd_info, main


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
