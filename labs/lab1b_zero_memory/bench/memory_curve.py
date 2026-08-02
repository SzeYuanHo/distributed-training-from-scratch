#!/usr/bin/env python3
"""Lab 1B deliverable: memory against world size, analytic and measured.

::

    python tasks.py bench LAB=1b

Draws per-rank memory against N for all four ZeRO stages (analytic, dashed) with
the measured points overlaid, and prints a reconciliation attributing the gap.

A caveat that matters for how you read the output. On CPU there is no equivalent
of ``torch.cuda.max_memory_allocated``, so the measured points come from process
RSS, which includes the python interpreter, torch itself, and the dataset. That is
tens of megabytes of constant offset on a model whose states are a few megabytes.
The *trend* against N is meaningful; the absolute value is not.

This sweep intentionally uses Gloo and CPU so its world-size trend is available
on an ordinary machine. Running it unchanged on a CUDA host still measures the
CPU process; a GPU allocator study must explicitly move the model, batches, and
optimizer to one CUDA device per rank.
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

from common import dist_utils, memory, reference_gpt, timers  # noqa: E402

WORLD_SIZES = [1, 2, 4, 8]
STAGES = [0, 1, 2, 3]


def _worker(preset: str, steps: int, use_zero: bool):
    from reference import ddp_baseline
    from starter import zero1

    rank, world = dist_utils.rank(), dist_utils.world_size()
    from common import data as data_mod

    dataset = data_mod.load_dataset()
    torch.manual_seed(1337)
    model = reference_gpt.make_model(preset, block_size=128, vocab_size=dataset.vocab_size)

    measurement_device = next(model.parameters()).device
    memory.reset_peak(measurement_device)

    if use_zero:
        dist_utils.sync_module_params(model)
        optimizer = zero1.ZeRO1Optimizer(model.parameters(), lr=1e-3)
        wrapper = model
        state_bytes_fn = optimizer.local_state_bytes
    else:
        wrapper = ddp_baseline.PlainDDP(model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
        state_bytes_fn = lambda: ddp_baseline.optimizer_state_bytes(optimizer)  # noqa: E731

    ddp_baseline.train_steps(wrapper, optimizer, steps, batch=8, block=128)

    peak = memory.measured_peak(measurement_device)
    return {
        "rank": rank,
        "world": world,
        "peak_allocated": peak["allocated"],
        "state_bytes": state_bytes_fn(),
        "n_params": model.num_params(),
    }


def plot(n_params: int, measured: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    colours = {0: "#2b6cb0", 1: "#dd6b20", 2: "#2c7a7b", 3: "#805ad5"}
    for stage in STAGES:
        analytic = [
            memory.analytic_ledger(n_params, "bf16", "adamw", stage, n).total / 1e6
            for n in WORLD_SIZES
        ]
        ax1.plot(
            WORLD_SIZES,
            analytic,
            "--o",
            ms=4,
            color=colours[stage],
            label=f"ZeRO-{stage} (analytic)",
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(WORLD_SIZES)
    ax1.set_xticklabels([str(n) for n in WORLD_SIZES])
    ax1.set_xlabel("world size")
    ax1.set_ylabel("per-rank states (MB)")
    ax1.set_title(f"Analytic ledger, {n_params / 1e6:.1f} M parameters")
    ax1.grid(alpha=0.25, ls=":")
    ax1.legend(fontsize=8)

    if measured.get("ddp") and measured.get("zero1"):
        worlds = sorted(measured["ddp"])
        ax2.plot(
            worlds,
            [measured["ddp"][w] / 1e6 for w in worlds],
            "-o",
            color=colours[0],
            label="DDP optimizer state (measured)",
        )
        ax2.plot(
            worlds,
            [measured["zero1"][w] / 1e6 for w in worlds],
            "-o",
            color=colours[1],
            label="ZeRO-1 optimizer state (measured)",
        )
        ideal = [measured["ddp"][worlds[0]] / w / 1e6 for w in worlds]
        ax2.plot(worlds, ideal, ":", color="#718096", label="perfect 1/N")

        ax2.set_xscale("log", base=2)
        ax2.set_xticks(worlds)
        ax2.set_xticklabels([str(w) for w in worlds])
        ax2.set_xlabel("world size")
        ax2.set_ylabel("optimizer state per rank (MB)")
        ax2.set_title("Measured optimizer state")
        ax2.grid(alpha=0.25, ls=":")
        ax2.legend(fontsize=8)
    else:
        ax2.text(
            0.5,
            0.5,
            "no measurements\n(is ZeRO1Optimizer implemented?)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_axis_off()

    fig.suptitle("Lab 1B: where the memory goes")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="small")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--nproc", type=int, default=None, help="single world size instead of {1,2,4}"
    )
    args = parser.parse_args(argv)

    import os

    worlds = [args.nproc] if args.nproc else [1, 2, 4]
    worlds = [w for w in worlds if w <= (os.cpu_count() or 4)]

    print(f"Lab 1B memory sweep, preset {args.preset}\n")

    measured = {"ddp": {}, "zero1": {}}
    n_params = 0

    for world in worlds:
        for label, use_zero in (("ddp", False), ("zero1", True)):
            try:
                results = dist_utils.run_distributed(
                    _worker,
                    world_size=world,
                    backend="gloo",
                    args=(args.preset, args.steps, use_zero),
                    timeout_s=400,
                )
            except NotImplementedError:
                print(f"  {label} at N={world}: ZeRO1Optimizer not implemented yet")
                continue

            n_params = results[0]["n_params"]
            state = max(r["state_bytes"] for r in results.values())
            peak = max(r["peak_allocated"] for r in results.values())
            measured[label][world] = state
            print(
                f"  {label:<6} N={world}   optimizer state {state / 1e6:>8.2f} MB   "
                f"process peak {peak / 1e6:>8.1f} MB"
            )

    if not n_params:
        print("\n  nothing measured. Finish starter/zero1.py first.")
        return 1

    print()
    analytic = memory.analytic_ledger(n_params, "bf16", "adamw", 0, 1)
    print(memory.reconcile(analytic, label=f"{args.preset} preset, N=1, ZeRO-0"))

    out_dir = timers.runs_dir("lab1b_zero_memory")
    image = plot(n_params, measured, out_dir / "memory_curve.png")

    summary = {
        "states_bytes_small_preset_n1": {
            "value": memory.analytic_ledger(n_params, "bf16", "adamw", 0, 1).total,
            "units": "bytes",
        },
        "states_bytes_small_preset_zero1_n4": {
            "value": memory.analytic_ledger(n_params, "bf16", "adamw", 1, 4).total,
            "units": "bytes",
        },
        "bytes_per_param_mixed_adamw": {"value": 16, "units": "bytes"},
    }
    if measured["ddp"] and measured["zero1"] and 4 in measured["zero1"]:
        summary["optimizer_state_reduction_n4"] = {
            "value": measured["ddp"][1] / measured["zero1"][4] if 1 in measured["ddp"] else 0,
            "units": "x",
        }

    (out_dir / "measurements.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=True), encoding="utf-8"
    )

    print(f"\n  plot         -> {image}")
    print(f"  measurements -> {out_dir / 'measurements.yaml'}")
    print("\n  next: python tasks.py reconcile LAB=1b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
