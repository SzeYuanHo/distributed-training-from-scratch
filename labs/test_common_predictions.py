"""Regression tests for the shared prediction/reconciliation report."""

from __future__ import annotations

from pathlib import Path

from common.predictions import Reconciler


def test_reconciliation_report_displays_units_once(tmp_path: Path) -> None:
    (tmp_path / "predictions.yaml").write_text(
        "wire_bytes:\n  value: 1024\n  units: bytes/rank/step\n",
        encoding="utf-8",
    )
    reconciler = Reconciler(tmp_path, title="unit contract")
    reconciler.measure("wire_bytes", 2048, units="bytes/rank/step")

    report = reconciler.report()

    assert "units" in report
    assert report.count("bytes/rank/step") == 1
    assert "1,024" in report
    assert "2,048" in report


def test_reconciliation_does_not_compare_incompatible_units(tmp_path: Path) -> None:
    (tmp_path / "predictions.yaml").write_text(
        "latency:\n  value: 1\n  units: seconds\n",
        encoding="utf-8",
    )
    reconciler = Reconciler(tmp_path, title="unit mismatch")
    reconciler.measure("latency", 1000, units="milliseconds")

    row = reconciler.rows()[0]
    report = reconciler.report()

    assert row["error"] is None
    assert row["band"] == "unit mismatch"
    assert "seconds != milliseconds" in report
    assert "Unit mismatches were not compared" in report
