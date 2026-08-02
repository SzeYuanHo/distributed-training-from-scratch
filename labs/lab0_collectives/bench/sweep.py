#!/usr/bin/env python3
"""Lab 0 deliverable: sweep message size and rank count, fit the cost model, plot it.

Runs your three all-reduce implementations across 2^10 to 2^26 bytes and world
sizes {2, 4, 8}, fits alpha and beta from the measurements, and draws measured
against predicted on log-log axes with the crossover marked.

::

    python tasks.py bench LAB=0
    python tasks.py bench LAB=0 NPROC=4

Writes ``runs/lab0_collectives/cost_model.png`` and
``runs/lab0_collectives/measurements.yaml``. The second is what
``python tasks.py reconcile LAB=0`` reads.

This fails until the starter code works. That is expected: the bench script is
instrumentation, not curriculum, and it is the last thing you run rather than
the first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_DIR.parent.parent
for path in (str(REPO_ROOT), str(LAB_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import repo_env  # noqa: E402

repo_env.apply()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import dist_utils, timers  # noqa: E402

SIZES_BYTES = [2**k for k in range(10, 27)]
ALGORITHMS = ["ring", "naive", "recursive_doubling"]
ITERS = 12
WARMUP = 3


def _median_seconds(fn, tensor: torch.Tensor) -> float:
    for _ in range(WARMUP):
        fn(tensor.clone())
    samples = []
    for _ in range(ITERS):
        work = tensor.clone()
        start = time.perf_counter()
        fn(work)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def _worker(sizes: list[int], algorithms: list[str]) -> dict:
    """Runs in every rank. Returns rank 0's timings; other ranks return None."""
    from starter import allreduce_algos

    impls = {
        "ring": allreduce_algos.all_reduce_ring,
        "naive": allreduce_algos.all_reduce_naive,
        "recursive_doubling": allreduce_algos.all_reduce_recursive_doubling,
    }

    rank, world = dist_utils.rank(), dist_utils.world_size()
    out: dict[str, dict[int, float]] = {}

    for name in algorithms:
        if name == "recursive_doubling" and (world & (world - 1)) != 0:
            continue  # needs a power-of-two world size
        out[name] = {}
        for n_bytes in sizes:
            tensor = torch.randn(n_bytes // 4, dtype=torch.float32) + rank
            try:
                out[name][n_bytes] = _median_seconds(impls[name], tensor)
            except NotImplementedError:
                dist_utils.printr(
                    f"{name} is not implemented yet; skipping the rest of its sweep",
                    only=0,
                )
                out[name] = {}
                break

    return out if rank == 0 else {}


def run_sweep(world_sizes: list[int], sizes: list[int]) -> dict:
    results: dict[int, dict] = {}
    for world in world_sizes:
        print(f"\n  world size {world}")
        collected = dist_utils.run_distributed(
            _worker,
            world_size=world,
            backend="gloo",
            args=(sizes, ALGORITHMS),
            timeout_s=600,
        )
        results[world] = collected.get(0, {})
        for name, points in results[world].items():
            if points:
                biggest = max(points)
                print(
                    f"    {name:<20} {len(points):>2} points, "
                    f"{biggest / 1e6:.0f} MB in {points[biggest] * 1e3:.2f} ms"
                )
    return results


def _fit_link_parameters(
    cost_model, algorithm: str, world: int, sizes: np.ndarray, times: np.ndarray
) -> tuple[float, float]:
    """Fit per-message latency and per-byte link cost from collective timings."""
    if world < 2:
        raise ValueError("link parameters require at least two ranks")
    if algorithm == "ring":
        latency_rounds = 2 * (world - 1)
        volume_factor = 2 * (world - 1) / world
    elif algorithm == "recursive_doubling":
        latency_rounds = 2 * np.log2(world)
        volume_factor = 2 * (world - 1) / world
    elif algorithm == "naive":
        latency_rounds = 2 * (world - 1)
        volume_factor = latency_rounds
    else:
        raise ValueError(f"unknown all-reduce algorithm {algorithm!r}")

    # fit_alpha_beta fits y = alpha + beta*x. Divide out the collective's
    # latency rounds and put its algorithmic byte factor into x, so alpha and
    # beta remain link-level parameters. Passing raw collective time/message
    # size here and then calling predict_allreduce would apply both factors twice.
    per_round_times = times / latency_rounds
    bytes_per_round = sizes * volume_factor / latency_rounds
    return cost_model.fit_alpha_beta(bytes_per_round, per_round_times)


def fit_and_report(results: dict) -> dict:
    """Fit the learner's cost model to the measurements and reconcile."""
    from starter import cost_model

    summary: dict[str, float] = {}

    for world, per_algorithm in results.items():
        if world < 2:
            continue
        ring = per_algorithm.get("ring", {})
        if len(ring) < 3:
            continue
        sizes = np.array(sorted(ring), dtype=np.float64)
        times = np.array([ring[int(s)] for s in sizes], dtype=np.float64)

        try:
            alpha, beta = _fit_link_parameters(cost_model, "ring", world, sizes, times)
        except NotImplementedError:
            print("\n  cost_model.fit_alpha_beta is not implemented; skipping the fit")
            return summary

        print(f"\n  fitted at world size {world}:")
        print(f"    alpha = {alpha:.3e} s        ({alpha * 1e6:.1f} us per message)")
        print(f"    beta  = {beta:.3e} s/byte    ({1 / beta / 1e9:.2f} GB/s effective)")

        if world == 4:
            summary["alpha_seconds"] = alpha
            summary["beta_seconds_per_byte"] = beta
            one_mb = 1 << 20
            if one_mb in ring:
                summary["allreduce_ring_1mb_n4_seconds"] = ring[one_mb]
            sixteen_mb = 1 << 24
            if sixteen_mb in ring:
                summary["allreduce_ring_16mb_n4_seconds"] = ring[sixteen_mb]
            summary["ring_allreduce_bytes_per_rank_n4_1mb"] = 2 * (world - 1) / world * one_mb

            rd = per_algorithm.get("recursive_doubling", {})
            if len(rd) >= 3:
                rd_sizes = np.array(sorted(rd), dtype=np.float64)
                rd_times = np.array([rd[int(s)] for s in rd_sizes], dtype=np.float64)
                _alpha_rd, beta_rd = _fit_link_parameters(
                    cost_model, "recursive_doubling", world, rd_sizes, rd_times
                )
                try:
                    summary["crossover_bytes_n4"] = cost_model.crossover_bytes(
                        world, alpha, beta, beta_rd
                    )
                except NotImplementedError:
                    pass

    return summary


def plot(results: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from starter import cost_model

    worlds = sorted(results)
    fig, axes = plt.subplots(1, len(worlds), figsize=(5 * len(worlds), 4.2), squeeze=False)

    colours = {"ring": "#2b6cb0", "naive": "#dd6b20", "recursive_doubling": "#2c7a7b"}

    for ax, world in zip(axes[0], worlds):
        per_algorithm = results[world]
        fits: dict[str, tuple[float, float]] = {}

        for name, points in per_algorithm.items():
            if not points:
                continue
            sizes = sorted(points)
            times = [points[s] for s in sizes]
            ax.loglog(
                sizes,
                [t * 1e3 for t in times],
                "o-",
                ms=3,
                color=colours.get(name, "#805ad5"),
                label=f"{name} measured",
            )

            if world >= 2 and name in {"ring", "recursive_doubling"} and len(sizes) >= 3:
                try:
                    fits[name] = _fit_link_parameters(
                        cost_model,
                        name,
                        world,
                        np.array(sizes, dtype=np.float64),
                        np.array(times, dtype=np.float64),
                    )
                except NotImplementedError:
                    pass

        if fits:
            grid = np.array(
                sorted(next(iter(per_algorithm.values())) or SIZES_BYTES), dtype=np.float64
            )
            for name in ("ring", "recursive_doubling"):
                if name not in fits:
                    continue
                if name == "recursive_doubling" and (world & (world - 1)):
                    continue
                alpha, beta = fits[name]
                try:
                    predicted = [
                        cost_model.predict_allreduce(name, world, int(s), alpha, beta) * 1e3
                        for s in grid
                    ]
                except (NotImplementedError, ValueError):
                    continue
                ax.loglog(
                    grid,
                    predicted,
                    "--",
                    lw=1.2,
                    color=colours.get(name, "#805ad5"),
                    alpha=0.7,
                    label=f"{name} predicted",
                )

        if {"ring", "recursive_doubling"} <= fits.keys():
            alpha_ring, beta_ring = fits["ring"]
            _alpha_rd, beta_rd = fits["recursive_doubling"]
            try:
                crossover = cost_model.crossover_bytes(world, alpha_ring, beta_ring, beta_rd)
            except (NotImplementedError, ValueError):
                crossover = math.inf
            if math.isfinite(crossover) and crossover > 0:
                ax.axvline(
                    crossover,
                    color="#805ad5",
                    linestyle=":",
                    linewidth=1.4,
                    label=f"fitted crossover ({crossover:,.0f} B)",
                )

        ax.set_title(f"world size {world}")
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("all-reduce time (ms)")
        ax.grid(True, which="both", alpha=0.25, linestyle=":")
        ax.legend(fontsize=7)

    fig.suptitle("Lab 0: measured against the fitted alpha-beta model (gloo, localhost)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="single world size instead of the full {2,4,8} sweep",
    )
    parser.add_argument(
        "--max-bytes", type=int, default=1 << 24, help="largest message to sweep (default 16 MB)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="four sizes instead of seventeen, for a fast sanity check",
    )
    args = parser.parse_args(argv)

    sizes = [s for s in SIZES_BYTES if s <= args.max_bytes]
    if args.quick:
        sizes = sizes[::4]

    world_sizes = [args.nproc] if args.nproc else [2, 4, 8]
    cores = __import__("os").cpu_count() or 4
    world_sizes = [w for w in world_sizes if w <= cores * 2]

    print("Lab 0 cost-model sweep")
    print(f"  world sizes {world_sizes}")
    print(f"  sizes       {len(sizes)} points, {sizes[0]} to {sizes[-1]} bytes")

    results = run_sweep(world_sizes, sizes)
    summary = fit_and_report(results)

    out_dir = timers.runs_dir("lab0_collectives")
    (out_dir / "raw_timings.json").write_text(
        json.dumps(
            {
                str(k): {a: {str(s): t for s, t in v.items()} for a, v in p.items()}
                for k, p in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if summary:
        import yaml

        (out_dir / "measurements.yaml").write_text(
            yaml.safe_dump({k: {"value": float(v)} for k, v in summary.items()}, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\n  measurements -> {out_dir / 'measurements.yaml'}")

    try:
        image = plot(results, out_dir / "cost_model.png")
        print(f"  plot         -> {image}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot failed: {type(exc).__name__}: {exc}")

    print("\n  next: python tasks.py reconcile LAB=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
