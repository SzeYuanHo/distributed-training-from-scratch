"""Timing, profiling, and the Gantt charts that make overlap visible.

Three tools, each for a different question.

``Timer``
    "How long did this region take?" Uses CUDA events on GPU and
    ``perf_counter`` on CPU, and hides the synchronisation difference so the
    same code works at Tier 1 and Tier 2.

``trace``
    "What kernels ran, and in what order?" A thin ``torch.profiler`` wrapper
    with a post-export size warning, because Chrome traces can grow into the
    hundreds of megabytes.

``EventLog``
    "Did communication actually overlap computation?" Records intervals across
    ranks and draws them as a Gantt chart. This is the one that matters at
    Tier 1, where a real profiler shows you a lot of CPU noise and very little
    insight. Lab 1 uses it for comm/compute overlap and Lab 3 for the pipeline
    schedule.

A note on clocks. Separate processes do not share a clock origin, and on
Windows ``perf_counter`` is not even guaranteed to have the same origin across
processes. ``EventLog`` handles this by taking a barrier-synchronised epoch at
construction: every rank barriers, then records its own ``perf_counter`` as
time zero. Residual skew is roughly the barrier latency, around a millisecond
on gloo/localhost. Good enough to see a 50 ms all-reduce overlapping a 200 ms
backward. Not good enough to measure a 100 microsecond kernel.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

__all__ = ["Timer", "trace", "EventLog", "Interval", "runs_dir"]


def runs_dir(lab: str | None = None) -> Path:
    """Where artefacts go. Honours ``DTFS_RUNS_DIR`` so tests can redirect it."""
    from common import repo_env

    base = os.environ.get("DTFS_RUNS_DIR", "runs")
    path = Path(base)
    if not path.is_absolute():
        path = repo_env.repo_root() / path
    if lab:
        path = path / lab
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Timer
# --------------------------------------------------------------------------- #


class Timer:
    """Region timer that does the right thing on both CPU and CUDA.

    ::

        timer = Timer()
        with timer("forward"):
            loss = model(x)
        with timer("backward"):
            loss.backward()
        print(timer.report())

    On CUDA it records ``torch.cuda.Event`` pairs and only synchronises when
    you ask for results, so timing does not serialise the stream you are trying
    to measure. On CPU it is ``perf_counter``. Either way ``elapsed("forward")``
    gives you milliseconds.
    """

    def __init__(self, device: torch.device | str | None = None, *, enabled: bool = True):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)
        self.enabled = enabled
        self._use_cuda = self.device.type == "cuda"
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._cuda_pairs: list[tuple[str, Any, Any]] = []

    @contextlib.contextmanager
    def __call__(self, label: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_pairs.append((label, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._samples[label].append((time.perf_counter() - t0) * 1e3)

    def _drain_cuda(self) -> None:
        if not self._cuda_pairs:
            return
        torch.cuda.synchronize()
        for label, start, end in self._cuda_pairs:
            self._samples[label].append(start.elapsed_time(end))
        self._cuda_pairs.clear()

    def elapsed(self, label: str) -> float:
        """Total milliseconds spent in ``label``."""
        self._drain_cuda()
        return sum(self._samples.get(label, []))

    def mean(self, label: str, skip_first: int = 0) -> float:
        """Mean milliseconds per call. Skip warmup iterations with ``skip_first``."""
        if skip_first < 0:
            raise ValueError(f"skip_first must be non-negative; got {skip_first}")
        self._drain_cuda()
        vals = self._samples.get(label, [])[skip_first:]
        return sum(vals) / len(vals) if vals else 0.0

    def counts(self) -> dict[str, int]:
        self._drain_cuda()
        return {k: len(v) for k, v in self._samples.items()}

    def as_dict(self) -> dict[str, dict[str, float]]:
        self._drain_cuda()
        out = {}
        for label, vals in self._samples.items():
            if not vals:
                continue
            ordered = sorted(vals)
            out[label] = {
                "calls": len(vals),
                "total_ms": sum(vals),
                "mean_ms": sum(vals) / len(vals),
                "min_ms": ordered[0],
                "median_ms": ordered[len(ordered) // 2],
                "max_ms": ordered[-1],
            }
        return out

    def report(self, skip_first: int = 0) -> str:
        if skip_first < 0:
            raise ValueError(f"skip_first must be non-negative; got {skip_first}")
        self._drain_cuda()
        if not self._samples:
            return "Timer: nothing recorded."
        lines = [
            f"Timer ({self.device.type}, {'cuda events' if self._use_cuda else 'perf_counter'})",
            "-" * 66,
            f"{'region':<24}{'calls':>7}{'total ms':>12}{'mean ms':>11}{'max ms':>11}",
        ]
        for label in sorted(self._samples, key=lambda k: -sum(self._samples[k])):
            vals = self._samples[label][skip_first:]
            total = sum(vals)
            mean = total / len(vals) if vals else 0.0
            maximum = max(vals) if vals else 0.0
            lines.append(f"{label:<24}{len(vals):>7}{total:>12.2f}{mean:>11.3f}{maximum:>11.3f}")
        return "\n".join(lines)

    def reset(self) -> None:
        self._samples.clear()
        self._cuda_pairs.clear()


# --------------------------------------------------------------------------- #
# profiler traces
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def trace(
    path: str | Path,
    *,
    active: int = 3,
    warmup: int = 1,
    wait: int = 0,
    record_shapes: bool = False,
    max_mb: int = 200,
) -> Iterator[Any]:
    """Emit a Chrome trace to ``path`` and warn when it exceeds ``max_mb``.

    ::

        with trace(runs_dir("lab1") / "step.json") as prof:
            for _ in range(10):
                train_step()
                prof.step()

    Open the result at ``chrome://tracing`` or https://ui.perfetto.dev.

    PyTorch exports the trace as one file, so ``max_mb`` is a warning threshold,
    not a hard byte cap. Three active steps is normally enough to see a pattern;
    use fewer steps or ``EventLog`` if the resulting trace is too large.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        record_shapes=record_shapes,
        profile_memory=True,
        with_stack=False,
    )
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.stop()
        profiler.export_chrome_trace(str(path))
        size_mb = path.stat().st_size / 1e6
        if size_mb > max_mb:
            print(
                f"warning: {path.name} is {size_mb:.0f} MB, over the {max_mb} MB threshold. "
                "Lower `active` or profile fewer steps."
            )
        else:
            print(f"trace written to {path} ({size_mb:.1f} MB)")


# --------------------------------------------------------------------------- #
# EventLog and Gantt
# --------------------------------------------------------------------------- #


@dataclass
class Interval:
    rank: int
    t_start: float  # seconds since this log's synchronised epoch
    t_end: float
    label: str
    category: str = "compute"  # compute | comm | idle | bubble

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class EventLog:
    """Append-only interval log across ranks, renderable as a Gantt chart.

    ::

        log = EventLog()
        with log.span("backward", category="compute"):
            loss.backward()
        with log.span("bucket 0 all-reduce", category="comm"):
            handle.wait()

        merged = log.gather()          # rank 0 gets every rank's intervals
        if dist_utils.is_main():
            log.plot_gantt(runs_dir("lab1") / "overlap.png", merged)

    The chart answers the question Lab 1 is actually about: are the comm bars
    sitting underneath the compute bars, or after them?
    """

    def __init__(self, *, sync_epoch: bool = True):
        self.intervals: list[Interval] = []
        self._epoch = self._synchronised_epoch() if sync_epoch else time.perf_counter()

    @staticmethod
    def _synchronised_epoch() -> float:
        """Barrier, then take the clock. Aligns rank origins to barrier latency."""
        from common import dist_utils

        if dist_utils.is_initialized():
            import torch.distributed as dist

            dist.barrier()
        return time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._epoch

    @contextlib.contextmanager
    def span(self, label: str, category: str = "compute") -> Iterator[None]:
        from common import dist_utils

        t0 = self.now()
        try:
            yield
        finally:
            self.intervals.append(Interval(dist_utils.rank(), t0, self.now(), label, category))

    def add(self, label: str, t_start: float, t_end: float, category: str = "compute") -> None:
        from common import dist_utils

        self.intervals.append(Interval(dist_utils.rank(), t_start, t_end, label, category))

    # -- aggregation -------------------------------------------------------- #

    def gather(self) -> list[Interval]:
        """Collect every rank's intervals onto every rank.

        Uses ``all_gather_object``, which goes through pickle and is therefore
        slow. Call it once at the end of a run, never inside the loop.
        """
        from common import dist_utils

        if not dist_utils.is_initialized() or dist_utils.world_size() == 1:
            return list(self.intervals)

        import torch.distributed as dist

        buckets: list[Any] = [None] * dist_utils.world_size()
        dist.all_gather_object(buckets, [asdict(i) for i in self.intervals])
        out: list[Interval] = []
        for chunk in buckets:
            out.extend(Interval(**d) for d in chunk or [])
        return out

    def occupancy(self, intervals: list[Interval] | None = None) -> dict[str, float]:
        """Fraction of the wall-clock span each category occupied, per rank averaged.

        ``comm_exposed`` is the interesting number: the share of time spent in
        communication with no compute running underneath it. Perfect overlap
        drives it to zero. In Lab 1 it never quite gets there, because the last
        bucket cannot start until the last gradient exists.
        """
        data = intervals if intervals is not None else self.intervals
        if not data:
            return {}

        span_start = min(i.t_start for i in data)
        span_end = max(i.t_end for i in data)
        total = max(span_end - span_start, 1e-12)

        by_rank: dict[int, list[Interval]] = defaultdict(list)
        for i in data:
            by_rank[i.rank].append(i)

        result: dict[str, float] = {}
        for category in {i.category for i in data}:
            share = 0.0
            for _rank, items in by_rank.items():
                share += sum(i.duration for i in items if i.category == category) / total
            result[category] = share / len(by_rank)

        # Exposed communication: comm time on a rank with no compute overlapping.
        exposed = 0.0
        for _rank, items in by_rank.items():
            comm = [i for i in items if i.category == "comm"]
            compute = [i for i in items if i.category == "compute"]
            for c in comm:
                covered = 0.0
                for k in compute:
                    lo, hi = max(c.t_start, k.t_start), min(c.t_end, k.t_end)
                    if hi > lo:
                        covered += hi - lo
                exposed += max(0.0, c.duration - covered)
        result["comm_exposed"] = (exposed / len(by_rank)) / total
        result["wall_s"] = total
        return result

    # -- output ------------------------------------------------------------- #

    def to_json(self, path: str | Path, intervals: list[Interval] | None = None) -> Path:
        data = intervals if intervals is not None else self.intervals
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(i) for i in data], indent=2), encoding="utf-8")
        return path

    def plot_gantt(
        self,
        path: str | Path,
        intervals: list[Interval] | None = None,
        *,
        title: str = "",
        figsize: tuple[float, float] | None = None,
    ) -> Path:
        """Draw one horizontal bar per interval, one row per rank."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        data = intervals if intervals is not None else self.intervals
        if not data:
            raise ValueError("EventLog is empty; nothing to plot.")

        ranks = sorted({i.rank for i in data})
        colours = {
            "compute": "#2b6cb0",
            "comm": "#dd6b20",
            "idle": "#a0aec0",
            "bubble": "#e2e8f0",
            "forward": "#2b6cb0",
            "backward": "#2c7a7b",
        }
        default_colour = "#805ad5"

        fig, ax = plt.subplots(figsize=figsize or (12, 1.1 * len(ranks) + 1.8))
        for interval in data:
            y = ranks.index(interval.rank)
            ax.barh(
                y,
                max(interval.duration, 1e-6),
                left=interval.t_start,
                height=0.6,
                color=colours.get(interval.category, default_colour),
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([f"rank {r}" for r in ranks])
        ax.invert_yaxis()
        ax.set_xlabel("seconds since synchronised epoch")
        ax.set_title(title or "Execution timeline")
        ax.grid(axis="x", alpha=0.25, linestyle=":")

        used = sorted({i.category for i in data})
        ax.legend(
            handles=[mpatches.Patch(color=colours.get(c, default_colour), label=c) for c in used],
            loc="upper right",
            fontsize=8,
        )

        occ = self.occupancy(data)
        if occ:
            ax.text(
                0.01,
                -0.22,
                f"wall {occ['wall_s'] * 1e3:.0f} ms   "
                f"compute {occ.get('compute', 0):.0%}   "
                f"comm {occ.get('comm', 0):.0%}   "
                f"exposed comm {occ.get('comm_exposed', 0):.0%}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return path
