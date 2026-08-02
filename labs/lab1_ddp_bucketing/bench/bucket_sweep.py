#!/usr/bin/env python3
"""Lab 1 deliverable 1: step time against bucket size, with your prediction marked.

::

    python tasks.py bench LAB=1

Sweeps {0.5, 1, 5, 10, 25, 100} MiB and plots step time, overlaid with the optimum
you predicted in ``predictions.yaml``. Writes ``runs/lab1_ddp_bucketing/``.

Read the resulting curve carefully. At Tier 1 it is usually flat-ish with a rise
at the small end, because gloo's per-message latency is high enough that the
latency term dominates everywhere and the tail term barely registers. That is a
real result about your machine, not a failed experiment, and the writeup should
say so with numbers.
"""

from __future__ import annotations

import argparse
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

import torch  # noqa: E402
import yaml  # noqa: E402

from common import comm_counter, dist_utils, predictions, timers  # noqa: E402

BUCKET_SIZES_MIB = [0.5, 1, 5, 10, 25, 100]
STEPS = 12
WARMUP = 3


def _worker(bucket_bytes: int, steps: int, preset: str):
    from reference import single_process as ref
    from starter import bucketed_ddp

    from common import data as data_mod

    rank, world = dist_utils.rank(), dist_utils.world_size()
    config = ref.TrainConfig(
        preset=preset, steps=steps, per_rank_batch=8, world_size=world, block_size=128
    )

    model = ref.build_model(config)
    wrapper = bucketed_ddp.BucketedDDP(model, bucket_bytes=bucket_bytes)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=config.lr)
    dataset = data_mod.load_dataset()

    timer = timers.Timer(device="cpu")
    counter = comm_counter.CommCounter()
    start = None

    with counter:
        for step in range(steps):
            x, y = data_mod.get_batch(
                "train",
                batch_size=config.per_rank_batch,
                block_size=config.block_size,
                rank=rank,
                world_size=world,
                step=step,
                dataset=dataset,
            )
            if step == WARMUP:
                start = time.perf_counter()
                counter.reset()

            with timer("forward"):
                _logits, loss = wrapper(x, y)
            optimizer.zero_grad(set_to_none=True)
            with timer("backward"):
                loss.backward()
            with timer("finalize"):
                wrapper.finalize()
            with timer("optimizer"):
                optimizer.step()

    elapsed = time.perf_counter() - start
    measured_steps = steps - WARMUP

    try:
        wrapper.close()
    except (AttributeError, NotImplementedError):
        pass

    return {
        "rank": rank,
        "step_ms": elapsed / measured_steps * 1e3,
        "collectives": counter.counts().get("all_reduce", 0) / measured_steps,
        "bytes_per_step": counter.total_bytes / measured_steps,
        "regions": timer.as_dict(),
    }


def _single_process_baseline(preset: str, steps: int) -> float:
    """Step time with no wrapper at all, for the honest comparison."""
    from reference import single_process as ref

    from common import data as data_mod

    config = ref.TrainConfig(
        preset=preset, steps=steps, per_rank_batch=8, world_size=1, block_size=128
    )
    model = ref.build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    dataset = data_mod.load_dataset()

    start = None
    for step in range(steps):
        x, y = data_mod.get_batch(
            "train",
            batch_size=config.per_rank_batch,
            block_size=config.block_size,
            rank=0,
            world_size=1,
            step=step,
            dataset=dataset,
        )
        if step == WARMUP:
            start = time.perf_counter()
        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return (time.perf_counter() - start) / (steps - WARMUP) * 1e3


def plot(
    sizes_mib: list[float],
    step_ms: list[float],
    collectives: list[float],
    predicted_mib: float | None,
    baseline_ms: float,
    out_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.semilogx(sizes_mib, step_ms, "o-", color="#2b6cb0", label="DDP step time")
    ax1.axhline(
        baseline_ms, ls="--", color="#718096", label=f"single process ({baseline_ms:.0f} ms)"
    )
    if predicted_mib:
        ax1.axvline(
            predicted_mib,
            ls=":",
            color="#dd6b20",
            lw=2,
            label=f"your prediction ({predicted_mib:.2g} MiB)",
        )
    best = sizes_mib[step_ms.index(min(step_ms))]
    ax1.axvline(
        best, ls="-", color="#2c7a7b", alpha=0.5, lw=2, label=f"measured best ({best:g} MiB)"
    )
    ax1.set_xlabel("bucket size (MiB)")
    ax1.set_ylabel("step time (ms)")
    ax1.set_title("Step time against bucket size")
    ax1.grid(True, which="both", alpha=0.25, ls=":")
    ax1.legend(fontsize=8)

    ax2.loglog(sizes_mib, collectives, "s-", color="#805ad5")
    ax2.set_xlabel("bucket size (MiB)")
    ax2.set_ylabel("all-reduces per step")
    ax2.set_title("Collective count against bucket size")
    ax2.grid(True, which="both", alpha=0.25, ls=":")

    fig.suptitle("Lab 1: the bucket-size trade-off (gloo, CPU)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args(argv)

    print("Lab 1 bucket-size sweep")
    print(f"  world size {args.nproc}, preset {args.preset}, {args.steps} steps\n")

    baseline = _single_process_baseline(args.preset, args.steps)
    print(f"  single-process baseline: {baseline:.1f} ms/step\n")

    sizes_mib, step_ms, collectives, bytes_per_step = [], [], [], []
    for mib in BUCKET_SIZES_MIB:
        n_bytes = int(mib * 1024 * 1024)
        try:
            results = dist_utils.run_distributed(
                _worker,
                world_size=args.nproc,
                backend="gloo",
                args=(n_bytes, args.steps, args.preset),
                timeout_s=600,
            )
        except NotImplementedError:
            print("  BucketedDDP is not implemented yet. Finish steps 5 and 6 first.")
            return 1

        worst = max(r["step_ms"] for r in results.values())
        sizes_mib.append(mib)
        step_ms.append(worst)
        collectives.append(results[0]["collectives"])
        bytes_per_step.append(results[0]["bytes_per_step"])
        print(
            f"  {mib:>6.1f} MiB  {worst:>8.1f} ms/step   "
            f"{results[0]['collectives']:>5.1f} all-reduces   "
            f"{results[0]['bytes_per_step'] / 1e6:>6.2f} MB/step"
        )

    best_mib = sizes_mib[step_ms.index(min(step_ms))]
    speedup = baseline / min(step_ms)

    print(f"\n  best measured bucket size: {best_mib:g} MiB")
    print(f"  speedup against single process: {speedup:.2f}x")
    if speedup < 1:
        print("  Below 1.0, as expected at Tier 1. Your ranks share one machine, so")
        print("  you added communication without adding compute. Explain the number")
        print("  in the writeup rather than treating it as a defect.")

    out_dir = timers.runs_dir("lab1_ddp_bucketing")

    reconciler = predictions.Reconciler(LAB_DIR)
    predicted = reconciler.predictions.get("optimal_bucket_bytes")
    predicted_mib = (predicted.value / 1024 / 1024) if predicted and predicted.stated else None

    measurements = {
        "optimal_bucket_bytes": {"value": best_mib * 1024 * 1024, "units": "bytes"},
        "optimal_bucket_count": {
            "value": collectives[sizes_mib.index(best_mib)],
            "units": "buckets",
        },
        "single_process_step_ms": {"value": baseline, "units": "ms"},
        f"ddp_{args.nproc}rank_step_ms": {"value": min(step_ms), "units": "ms"},
        "ddp_2rank_step_ms": {"value": min(step_ms), "units": "ms"},
        "tier1_speedup_vs_single_process": {"value": speedup, "units": "x"},
        "bucketed_allreduces_per_step_1mb": {
            "value": collectives[sizes_mib.index(1)] if 1 in sizes_mib else 0,
            "units": "collectives",
        },
    }
    (out_dir / "measurements.yaml").write_text(
        yaml.safe_dump(measurements, sort_keys=True), encoding="utf-8"
    )

    image = plot(
        sizes_mib,
        step_ms,
        collectives,
        predicted_mib,
        baseline,
        out_dir / "bucket_sweep.png",
    )

    print(f"\n  plot         -> {image}")
    print(f"  measurements -> {out_dir / 'measurements.yaml'}")
    print("\n  next: python tasks.py gantt LAB=1")
    print("        python tasks.py reconcile LAB=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
