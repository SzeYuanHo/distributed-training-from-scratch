"""The predict-measure-reconcile harness.

Design principle P3: before you run a lab, you write down what you think will
happen. Bytes on the wire, peak memory, bubble fraction, step time. Then the
lab measures the real value and prints both.

Being wrong is the point. A prediction that misses by 4x is a specific, named
gap in your model of the machine, and chasing it down teaches more than the
implementation did. A prediction you never wrote down teaches nothing, because
after the fact every number looks reasonable.

This module never fails a test. It is an instrument, not a gate. A lab whose
tests pass and whose predictions were all wrong is a lab you finished and did
not understand, and that distinction should be visible rather than enforced.

``predictions.yaml`` looks like this::

    ring_allreduce_bytes_per_rank:
      value: 786432
      units: bytes
      note: 2*(N-1)/N * M for N=4, M=512KB

    alpha_seconds:
      value: 3.0e-5
      units: s
      note: guessed from loopback ping

Every entry needs ``value``. ``units`` and ``note`` are optional and both show
up in the report. Leaving ``value: null`` marks it as not yet predicted, and
the reconciliation says so instead of scoring it.

Some entries are analytic or code-inspection checks rather than runtime
measurements. They can remain without a recorded counterpart; the report does
not imply that every such entry is emitted by a benchmark.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Prediction", "Reconciler", "load_predictions", "main"]

# Error bands. Wide on purpose: predicting a collective's latency within 10%
# from first principles is genuinely hard, and a 40% miss still means you had
# the right model.
GREEN_BAND = 0.10
YELLOW_BAND = 0.50


@dataclass
class Prediction:
    key: str
    value: float | None
    units: str = ""
    note: str = ""

    @property
    def stated(self) -> bool:
        return self.value is not None


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    """Read a ``predictions.yaml``. Missing file gives an empty dict, not an error."""
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.yaml"
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, Prediction] = {}
    for key, entry in raw.items():
        if isinstance(entry, dict):
            out[key] = Prediction(
                key=key,
                value=_as_float(entry.get("value")),
                units=str(entry.get("units", "")),
                note=str(entry.get("note", "")),
            )
        else:
            out[key] = Prediction(key=key, value=_as_float(entry))
    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, units: str = "", *, missing: str = "not predicted") -> str:
    if value is None:
        return missing
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
        text = f"{value:.4g}"
    elif float(value).is_integer():
        text = f"{int(value):,}"
    else:
        text = f"{value:,.4g}"
    return f"{text} {units}".strip()


class Reconciler:
    """Collects measurements and prints them against the predictions.

    ::

        rec = Reconciler(Path(__file__).parent.parent)   # the lab directory
        rec.measure("ring_allreduce_bytes_per_rank", counter.total_bytes, units="bytes")
        rec.measure("alpha_seconds", alpha, units="s")
        print(rec.report())
        rec.save()          # writes runs/labN/reconciliation.md
    """

    def __init__(self, lab_dir: str | Path, *, title: str = ""):
        self.lab_dir = Path(lab_dir)
        self.title = title or self.lab_dir.name
        self.predictions = load_predictions(self.lab_dir)
        self.measurements: dict[str, tuple[float, str]] = {}

    def measure(self, key: str, value: float, units: str = "") -> float:
        """Record a measured value. Returns it, so you can inline the call."""
        self.measurements[key] = (float(value), units)
        return value

    def rows(self) -> list[dict[str, Any]]:
        keys = list(dict.fromkeys([*self.predictions, *self.measurements]))
        out = []
        for key in keys:
            pred = self.predictions.get(key)
            measured, m_units = self.measurements.get(key, (None, ""))
            unit_mismatch = bool(pred and pred.units and m_units and pred.units != m_units)
            units = (
                f"{pred.units} != {m_units}"
                if unit_mismatch and pred
                else (pred.units if pred and pred.units else m_units)
            )

            error: float | None = None
            band = "-"
            if unit_mismatch:
                band = "unit mismatch"
            elif pred and pred.stated and measured is not None:
                if pred.value == 0:
                    error = 0.0 if measured == 0 else math.inf
                else:
                    error = abs(measured - pred.value) / abs(pred.value)
                band = (
                    "ok" if error <= GREEN_BAND else ("close" if error <= YELLOW_BAND else "MISS")
                )
            elif measured is None:
                band = "unmeasured"
            elif not pred or not pred.stated:
                band = "unpredicted"

            out.append(
                {
                    "key": key,
                    "predicted": pred.value if pred else None,
                    "measured": measured,
                    "units": units,
                    "error": error,
                    "band": band,
                    "note": pred.note if pred else "",
                }
            )
        return out

    def report(self) -> str:
        rows = self.rows()
        if not rows:
            return (
                f"{self.title}: nothing to reconcile.\n"
                f"  Fill in {self.lab_dir / 'predictions.yaml'} and run the lab's bench script."
            )

        width = max(len(r["key"]) for r in rows) + 2
        units_width = max(7, *(len(r["units"]) for r in rows)) + 2
        rule_width = width + 16 + 16 + units_width + 10 + 8
        lines = [
            f"Predictions vs measurements: {self.title}",
            "=" * rule_width,
            f"{'quantity':<{width}}"
            f"{'predicted':>16}"
            f"{'measured':>16}"
            f"{'units':>{units_width}}"
            f"{'error':>10}"
            f"{'':>8}",
            "-" * rule_width,
        ]
        for r in rows:
            err = (
                ""
                if r["error"] is None
                else ("inf" if math.isinf(r["error"]) else f"{r['error'] * 100:>8.1f}%")
            )
            lines.append(
                f"{r['key']:<{width}}"
                f"{_fmt(r['predicted']):>16}"
                f"{_fmt(r['measured'], missing='not recorded'):>16}"
                f"{r['units']:>{units_width}}"
                f"{err:>10}"
                f"  {r['band']}"
            )
        lines.append("-" * rule_width)

        misses = [r for r in rows if r["band"] == "MISS"]
        unpredicted = [r for r in rows if r["band"] == "unpredicted"]
        unmeasured = [r for r in rows if r["band"] == "unmeasured"]
        unit_mismatches = [r for r in rows if r["band"] == "unit mismatch"]

        if misses:
            lines += ["", "Off by more than 50%. These are the ones worth understanding:"]
            for r in misses:
                lines.append(
                    f"  {r['key']}" + (f"   (you wrote: {r['note']})" if r["note"] else "")
                )
            lines.append("  Write down why in runs/<lab>/writeup.md before moving on.")
        if unpredicted:
            lines += [
                "",
                "Measured but never predicted: " + ", ".join(r["key"] for r in unpredicted),
            ]
        if unmeasured:
            lines += [
                "",
                "Predictions without a recorded counterpart: "
                + ", ".join(r["key"] for r in unmeasured),
                "  Some are analytic or inspection results. Check the relevant "
                "derivation or test; rerun a benchmark only for quantities it records.",
            ]
        if unit_mismatches:
            lines += [
                "",
                "Unit mismatches were not compared: "
                + ", ".join(f"{r['key']} ({r['units']})" for r in unit_mismatches),
            ]
        if not misses and not unpredicted and not unmeasured and not unit_mismatches:
            lines += ["", "Everything predicted and everything within 50%. Good calibration."]

        return "\n".join(lines)

    def save(self, path: str | Path | None = None) -> Path:
        from common.timers import runs_dir

        if path is None:
            path = runs_dir(self.lab_dir.name) / "reconciliation.md"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("```\n" + self.report() + "\n```\n", encoding="utf-8")
        return path


def main(argv: list[str] | None = None) -> int:
    """``python -m common.predictions --lab labs/lab0_collectives``.

    Reads the lab's ``predictions.yaml`` and whatever the last bench run wrote
    to ``runs/<lab>/measurements.yaml``, then prints the table. ``tasks.py
    reconcile`` calls this.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--lab", required=True, help="path to a labs/labN_* directory")
    parser.add_argument("--measurements", default=None, help="override the measurements yaml path")
    args = parser.parse_args(argv)

    lab_dir = Path(args.lab)
    rec = Reconciler(lab_dir)

    from common.timers import runs_dir

    m_path = (
        Path(args.measurements)
        if args.measurements
        else runs_dir(lab_dir.name) / "measurements.yaml"
    )

    if m_path.exists():
        data = yaml.safe_load(m_path.read_text(encoding="utf-8")) or {}
        for key, entry in data.items():
            if isinstance(entry, dict):
                rec.measure(key, entry.get("value", 0), entry.get("units", ""))
            else:
                rec.measure(key, entry)
    else:
        print(f"note: no measurements at {m_path}. Run `python tasks.py bench LAB=...` first.\n")

    print(rec.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
