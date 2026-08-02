#!/usr/bin/env python3
"""Lab 3 deliverable 2: measured idle fraction against pipeline theory.

::

    python tasks.py bench LAB=3

Sweeps ``m in {1,2,4,8,16}`` for ``p in {2,4}`` and plots the measured idle
fraction against ``(p-1)/(m+p-1)``. The related overhead ratio relative to
ideal work is ``(p-1)/m``; the two have different denominators.

The measurement will sit above the curve, and by how much is the interesting part.
Three things the model leaves out, and the writeup asks you to attribute the gap
between them:

* **Stage imbalance.** The formula assumes every stage takes the same time. Yours
  does not, because the first stage carries the embedding and the last carries the
  LM head and the loss.
* **Point-to-point latency.** The formula counts idle *slots*, not the microseconds
  spent moving an activation between processes.
* **Python.** One interpreter frame per micro-batch per stage, which at these
  model sizes is not negligible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_DIR.parent.parent
for path in (str(REPO_ROOT), str(LAB_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import repo_env  # noqa: E402

repo_env.apply()

import yaml  # noqa: E402

from common import dist_utils, timers  # noqa: E402

MICRO_BATCHES = [1, 2, 4, 8, 16]
STAGE_COUNTS = [2, 4]


def theoretical_idle_fraction(stages: int, micro_batches: int) -> float:
    return (stages - 1) / (micro_batches + stages - 1)


def _worker(kind: str, n_micro: int, n_layer: int):
    from reference import single_process as ref
    from starter import partition, schedule_1f1b, schedule_afab

    config = ref.PipeConfig(n_layer=n_layer, n_micro=n_micro, batch_size=16)
    model = ref.build_model(config)
    stage = partition.split_model(model, dist_utils.world_size())[dist_utils.rank()]

    x, y = ref.get_batch(config)
    micro = config.micro_batch_size
    xs = [x[i * micro : (i + 1) * micro] for i in range(n_micro)]
    ys = [y[i * micro : (i + 1) * micro] for i in range(n_micro)]

    runner = schedule_afab.run_afab if kind == "afab" else schedule_1f1b.run_1f1b
    runner(stage, n_micro, xs, ys)  # warmup

    log = timers.EventLog()
    runner(stage, n_micro, xs, ys, event_log=log)
    merged = log.gather()

    if dist_utils.rank() == 0:
        return {"occupancy": log.occupancy(merged)}
    return {}


def plot(data: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colours = {2: "#2b6cb0", 4: "#dd6b20"}

    for p, points in sorted(data.items()):
        if not points:
            continue
        ms = sorted(points)
        measured = [points[m] for m in ms]
        theory = [theoretical_idle_fraction(p, m) for m in ms]

        ax.plot(ms, measured, "o-", color=colours.get(p, "#805ad5"), label=f"p={p} measured")
        ax.plot(
            ms,
            theory,
            "--",
            color=colours.get(p, "#805ad5"),
            alpha=0.6,
            label=f"p={p} theory (p-1)/(m+p-1)",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(MICRO_BATCHES)
    ax.set_xticklabels([str(m) for m in MICRO_BATCHES])
    ax.set_xlabel("micro-batches m")
    ax.set_ylabel("bubble fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Lab 3: pipeline bubble, measured against theory (1F1B, gloo)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)

    ax.text(
        0.02,
        0.02,
        "Measured sits above theory. The model counts idle slots and ignores\n"
        "stage imbalance, point-to-point latency, and python overhead.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="#4a5568",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nproc", type=int, default=None, help="single stage count instead of {2,4}"
    )
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kind", default="1f1b", choices=["1f1b", "afab"])
    args = parser.parse_args(argv)

    stage_counts = [args.nproc] if args.nproc else STAGE_COUNTS
    cores = os.cpu_count() or 4
    stage_counts = [p for p in stage_counts if p <= cores]

    print(f"Lab 3 bubble sweep ({args.kind})\n")
    data: dict[int, dict[int, float]] = {}

    for p in stage_counts:
        data[p] = {}
        for m in MICRO_BATCHES:
            if m < 1:
                continue
            try:
                results = dist_utils.run_distributed(
                    _worker,
                    world_size=p,
                    backend="gloo",
                    args=(args.kind, m, args.layers),
                    timeout_s=300,
                    join_timeout_s=330,
                )
            except NotImplementedError:
                print("  schedules not implemented yet; finish steps 3 and 5 first")
                return 1
            except Exception as exc:  # noqa: BLE001
                print(f"  p={p} m={m}: {type(exc).__name__}: {exc}")
                continue

            occ = results.get(0, {}).get("occupancy", {})
            bubble = max(0.0, 1.0 - occ.get("compute", 0.0))
            data[p][m] = bubble
            print(
                f"  p={p} m={m:>2}   measured {bubble:.3f}   "
                f"theory {theoretical_idle_fraction(p, m):.3f}   "
                f"excess {bubble - theoretical_idle_fraction(p, m):+.3f}"
            )

    out_dir = timers.runs_dir("lab3_pipeline_parallel")
    image = plot(data, out_dir / "bubble_sweep.png")

    summary = {}
    for p, points in data.items():
        for m, bubble in points.items():
            summary[f"theoretical_bubble_p{p}_m{m}"] = {
                "value": theoretical_idle_fraction(p, m),
                "units": "fraction",
            }
            summary[f"measured_bubble_p{p}_m{m}"] = {"value": bubble, "units": "fraction"}

    if summary:
        existing = out_dir / "measurements.yaml"
        merged = {}
        if existing.exists():
            merged = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        merged.update(summary)
        existing.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")

    print(f"\n  plot -> {image}")
    print("\n  next: python tasks.py reconcile LAB=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
