#!/usr/bin/env python3
"""Lab 3's headline deliverable: the two schedules, side by side.

::

    python tasks.py gantt LAB=3

Runs AFAB and 1F1B on the same partition with the same micro-batches, records
every forward and backward interval per stage, and draws both charts.

What to look for. The AFAB chart shows a long block of forwards on every stage,
then a long block of backwards, with a visible triangle of idle time at each end.
The 1F1B chart shows the same total work compressed into an interleaved pattern,
with the same triangles. **The bubble is identical. The activation memory is not.**
That is the entire lesson and the picture makes it in two seconds.
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

from common import dist_utils, memory, timers  # noqa: E402


def _worker(kind: str, n_micro: int, n_layer: int):
    from reference import single_process as ref
    from starter import partition, schedule_1f1b, schedule_afab

    config = ref.PipeConfig(n_layer=n_layer, n_micro=n_micro, batch_size=8)
    model = ref.build_model(config)
    stage = partition.split_model(model, dist_utils.world_size())[dist_utils.rank()]

    x, y = ref.get_batch(config)
    micro = config.micro_batch_size
    xs = [x[i * micro : (i + 1) * micro] for i in range(n_micro)]
    ys = [y[i * micro : (i + 1) * micro] for i in range(n_micro)]

    runner = schedule_afab.run_afab if kind == "afab" else schedule_1f1b.run_1f1b

    runner(stage, n_micro, xs, ys)  # warmup, not recorded

    measurement_device = next(stage.parameters()).device
    memory.reset_peak(measurement_device)
    log = timers.EventLog()
    loss = runner(stage, n_micro, xs, ys, event_log=log)
    peak = memory.measured_peak(measurement_device)

    merged = log.gather()
    if dist_utils.rank() == 0:
        return {
            "intervals": [i.__dict__ for i in merged],
            "occupancy": log.occupancy(merged),
            "loss": float(loss),
            "peak": peak["allocated"],
        }
    return {"peak": peak["allocated"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nproc", type=int, default=4, help="pipeline stages")
    parser.add_argument("--micro", type=int, default=8, help="micro-batches")
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args(argv)

    out_dir = timers.runs_dir("lab3_pipeline_parallel")
    overhead_ratio = (args.nproc - 1) / args.micro
    theory = (args.nproc - 1) / (args.micro + args.nproc - 1)
    summary: dict[str, dict] = {}

    print(f"Lab 3 schedule charts: p={args.nproc}, m={args.micro}")
    print(f"  theoretical idle fraction (p-1)/(m+p-1) = {theory:.3f}")
    print(f"  overhead relative to ideal work (p-1)/m = {overhead_ratio:.3f}\n")

    for kind, label in (("afab", "AFAB (GPipe)"), ("1f1b", "1F1B")):
        try:
            results = dist_utils.run_distributed(
                _worker,
                world_size=args.nproc,
                backend="gloo",
                args=(kind, args.micro, args.layers),
                timeout_s=400,
                join_timeout_s=440,
            )
        except NotImplementedError:
            print(f"  {label}: schedule not implemented yet")
            continue

        payload = results.get(0, {})
        if "intervals" not in payload:
            print(f"  {label}: rank 0 returned no intervals")
            continue

        intervals = [timers.Interval(**d) for d in payload["intervals"]]
        log = timers.EventLog(sync_epoch=False)
        image = log.plot_gantt(
            out_dir / f"schedule_{kind}.png",
            intervals,
            title=f"Lab 3: {label}   p={args.nproc}, m={args.micro}   theory bubble {theory:.3f}",
        )

        occ = payload["occupancy"]
        busy = occ.get("compute", 0.0)
        bubble = max(0.0, 1.0 - busy)
        peak = max(r.get("peak", 0) for r in results.values())

        summary[f"measured_bubble_{kind}_p{args.nproc}_m{args.micro}"] = {
            "value": bubble,
            "units": "fraction",
        }
        summary[f"peak_memory_{kind}"] = {"value": peak, "units": "bytes"}

        print(f"  {label}")
        print(f"    wall            {occ.get('wall_s', 0) * 1e3:8.0f} ms")
        print(f"    stages busy     {busy:8.1%}")
        print(f"    measured bubble {bubble:8.3f}   (theory {theory:.3f})")
        print(f"    peak memory     {peak / 1e6:8.1f} MB")
        print(f"    -> {image}")
        print()

    if summary:
        key = f"measured_bubble_p{args.nproc}_m{args.micro}"
        one = summary.get(f"measured_bubble_1f1b_p{args.nproc}_m{args.micro}")
        if one:
            summary[key] = one
        summary[f"theoretical_bubble_p{args.nproc}_m{args.micro}"] = {
            "value": theory,
            "units": "fraction",
        }

        existing = out_dir / "measurements.yaml"
        merged = {}
        if existing.exists():
            merged = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        merged.update(summary)
        existing.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
        print(f"  measurements -> {existing}")

    print("\n  Put the two charts side by side in your writeup. One thing changes")
    print("  between them and one thing does not; say which.")
    print("\n  next: python tasks.py bench LAB=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
