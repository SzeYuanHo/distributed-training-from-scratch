#!/usr/bin/env python3
"""Lab 1 deliverable 2: the overlap chart.

::

    python tasks.py gantt LAB=1

Records compute and communication intervals per rank during a few training steps
and draws them as a Gantt chart. One question: are orange communication requests
in flight underneath blue compute bars, or do they interrupt backward compute?

Runs twice, with overlap on and off, so you can see the difference rather than
take it on faith.

Reading it at Tier 1 needs a caveat. On CPU the compute is fast and gloo's
per-message latency is high, so the communication bars are wide and the picture
looks worse than a GPU cluster's would. What the chart still shows correctly is
*ordering*: whether a bucket's collective was issued as soon as its gradients
were ready, or held until the backward pass finished. That is the thing you
implemented, and it is visible regardless of the absolute times.

For an asynchronous bucket, an orange bar spans from collective issuance until
the explicit wait returns. That is the observable in-flight lifetime, not a claim
that the network link was busy for every instant in the interval.
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

import torch  # noqa: E402
import yaml  # noqa: E402

from common import dist_utils, timers  # noqa: E402

STEPS = 6
WARMUP = 2


def _worker(overlap: bool, bucket_bytes: int, preset: str):
    from reference import single_process as ref
    from starter import bucketed_ddp

    from common import collectives
    from common import data as data_mod

    rank, world = dist_utils.rank(), dist_utils.world_size()
    config = ref.TrainConfig(
        preset=preset, steps=STEPS, per_rank_batch=8, world_size=world, block_size=128
    )

    model = ref.build_model(config)
    wrapper = bucketed_ddp.BucketedDDP(model, bucket_bytes=bucket_bytes, overlap=overlap)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=config.lr)
    dataset = data_mod.load_dataset()

    log = timers.EventLog()
    original_all_reduce = collectives.all_reduce
    recording = False
    collective_index = 0

    class _TimedHandle:
        def __init__(self, handle, label: str, started: float):
            self._handle = handle
            self._label = label
            self._started = started
            self._logged = False

        def wait(self, timeout=None):
            result = self._handle.wait(timeout)
            if not self._logged:
                log.add(self._label, self._started, log.now(), "comm")
                self._logged = True
            return result

        def is_completed(self):
            return self._handle.is_completed()

        @property
        def waited(self):
            return self._handle.waited

    def timed_all_reduce(*args, **kwargs):
        nonlocal collective_index
        if not recording:
            return original_all_reduce(*args, **kwargs)

        label = f"all-reduce {collective_index}"
        collective_index += 1
        started = log.now()
        handle = original_all_reduce(*args, **kwargs)
        if kwargs.get("async_op", False) and handle is not None:
            return _TimedHandle(handle, label, started)
        log.add(label, started, log.now(), "comm")
        return handle

    collectives.all_reduce = timed_all_reduce
    try:
        for step in range(STEPS):
            x, y = data_mod.get_batch(
                "train",
                batch_size=config.per_rank_batch,
                block_size=config.block_size,
                rank=rank,
                world_size=world,
                step=step,
                dataset=dataset,
            )
            recording = step >= WARMUP

            if recording:
                with log.span(f"fwd {step}", "compute"):
                    _logits, loss = wrapper(x, y)
                optimizer.zero_grad(set_to_none=True)
                with log.span(f"bwd {step}", "compute"):
                    loss.backward()
                wrapper.finalize()
                optimizer.step()
            else:
                _logits, loss = wrapper(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.finalize()
                optimizer.step()
    finally:
        collectives.all_reduce = original_all_reduce

    if not overlap:
        # A synchronous hook blocks autograd, so the enclosing backward span
        # includes communication wall time. Split that blue span around each
        # orange interval instead of drawing the same time as both compute and
        # communication.
        blocking = sorted(
            (interval for interval in log.intervals if interval.category == "comm"),
            key=lambda interval: interval.t_start,
        )
        adjusted: list[timers.Interval] = []
        for interval in log.intervals:
            if interval.category != "compute":
                adjusted.append(interval)
                continue

            cursor = interval.t_start
            for block in blocking:
                start = max(interval.t_start, block.t_start)
                end = min(interval.t_end, block.t_end)
                if end <= start:
                    continue
                if start > cursor:
                    adjusted.append(
                        timers.Interval(
                            rank=interval.rank,
                            t_start=cursor,
                            t_end=start,
                            label=interval.label,
                            category=interval.category,
                        )
                    )
                cursor = max(cursor, end)
            if cursor < interval.t_end:
                adjusted.append(
                    timers.Interval(
                        rank=interval.rank,
                        t_start=cursor,
                        t_end=interval.t_end,
                        label=interval.label,
                        category=interval.category,
                    )
                )
        log.intervals = adjusted

    merged = log.gather()
    occupancy = log.occupancy(merged)

    if rank == 0:
        return {"intervals": [i.__dict__ for i in merged], "occupancy": occupancy}
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--bucket-mb", type=float, default=1.0)
    args = parser.parse_args(argv)

    out_dir = timers.runs_dir("lab1_ddp_bucketing")
    bucket_bytes = int(args.bucket_mb * 1024 * 1024)
    summary = {}

    for overlap in (True, False):
        label = "overlap" if overlap else "serial"
        print(f"\n  recording with overlap={overlap} ...")

        try:
            results = dist_utils.run_distributed(
                _worker,
                world_size=args.nproc,
                backend="gloo",
                args=(overlap, bucket_bytes, args.preset),
                timeout_s=600,
            )
        except NotImplementedError:
            print("  BucketedDDP is not implemented yet. Finish steps 5 and 6 first.")
            return 1

        payload = results.get(0, {})
        if not payload:
            print("  rank 0 returned nothing; check the worker")
            continue

        intervals = [timers.Interval(**d) for d in payload["intervals"]]
        # Draw compute first and communication second so nested orange bars are
        # visible instead of being covered by the enclosing backward interval.
        intervals.sort(key=lambda interval: (interval.category != "compute", interval.t_start))
        log = timers.EventLog(sync_epoch=False)
        image = log.plot_gantt(
            out_dir / f"overlap_gantt_{label}.png",
            intervals,
            title=f"Lab 1: compute and communication, overlap={overlap} ({args.nproc} gloo ranks)",
        )

        occ = payload["occupancy"]
        summary[f"exposed_comm_fraction_{label}"] = {
            "value": occ.get("comm_exposed", 0.0),
            "units": "fraction",
        }
        print(
            f"    wall {occ.get('wall_s', 0) * 1e3:.0f} ms   "
            f"compute {occ.get('compute', 0):.0%}   "
            f"comm {occ.get('comm', 0):.0%}   "
            f"exposed comm {occ.get('comm_exposed', 0):.0%}"
        )
        print(f"    -> {image}")

    if summary:
        summary["exposed_comm_fraction"] = summary.get(
            "exposed_comm_fraction_overlap", {"value": 0.0, "units": "fraction"}
        )
        existing = out_dir / "measurements.yaml"
        merged = {}
        if existing.exists():
            merged = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        merged.update(summary)
        existing.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")

    print("\n  Compare the two charts. With overlap on, communication requests stay")
    print("  in flight under compute. With it off, each orange bar interrupts backward.")
    print("\n  next: python tasks.py reconcile LAB=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
