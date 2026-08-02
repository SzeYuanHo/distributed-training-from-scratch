#!/usr/bin/env python3
"""Lab 2 deliverable: logical collective payload against tp_size, measured and predicted.

::

    python tasks.py bench LAB=2

Two panels. The left one shows collective tensor payload bytes counted per step
against tp_size, with the `b·s·h` prediction overlaid. It should be close to
flat: the all-reduced tensor is `(b, s, h)` no matter how many ranks share the
work. This counter value is not topology-dependent link traffic.

The right one is the point of the lab. It plots the compute-to-communication ratio
against tp_size for two hardware profiles, using the α and β you fitted in Lab 0
for gloo and the published numbers in ``reference/cluster_specs.py`` for T4 and
A100. The crossing point is where tensor parallelism stops paying, and it is why
TP lives inside an NVLink domain.

The right panel is a *derivation*, not a measurement. You do not have eight A100s.
Producing a defensible number without the hardware is the skill.
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

from common import comm_counter, dist_utils, timers  # noqa: E402

TP_SIZES = [2, 4]
BATCH, SEQ = 4, 64


def _worker(n_layer: int, hidden: int, batch: int, seq: int):
    from reference import plain_blocks
    from starter import tp_gpt

    rank, world = dist_utils.rank(), dist_utils.world_size()
    config = plain_blocks.tiny_config(
        n_layer=n_layer, n_embd=hidden, n_head=8, block_size=seq, vocab_size=64, max_tp=8
    )
    reference = plain_blocks.make_model(config, seed=0)
    parallel = tp_gpt.TensorParallelGPT.from_reference(reference)

    ids = torch.randint(0, config.vocab_size, (batch, seq))
    targets = torch.randint(0, config.vocab_size, (batch, seq))

    with comm_counter.CommCounter() as counter:
        _logits, loss = parallel(ids, targets)
        loss.backward()

    return {
        "rank": rank,
        "world": world,
        "bytes": counter.total_bytes,
        "counts": counter.counts(),
        "n_layer": n_layer,
        "hidden": hidden,
    }


def predicted_bytes(n_layer: int, hidden: int, batch: int, seq: int, elem_bytes: int = 4) -> int:
    """Logical payload for 4 all-reduces/layer/step on ``(b, s, h)`` tensors."""
    return 4 * n_layer * batch * seq * hidden * elem_bytes


def ceiling_curve(
    hidden: int, tokens: int, peak_flops: float, bandwidth: float, alpha: float, elem_bytes: int = 2
):
    """Compute time and communication time per layer against t.

    Compute per layer, sharded t ways, is roughly ``24·b·s·h²/t`` FLOPs.
    Communication is 2 all-reduces of ``b·s·h`` elements in the forward pass, at
    ring cost ``2(t-1)α + 2(t-1)/t · M · β``.
    """
    if peak_flops <= 0:
        raise ValueError("peak_flops must be positive for a compute ceiling")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive for a communication ceiling")

    # t=1 has no TP communication, so compute/communication is undefined.
    ts, compute, comm = [], [], []
    for t in (2, 4, 8, 16, 32, 64):
        flops = 24 * tokens * hidden * hidden / t
        message = tokens * hidden * elem_bytes
        ring = 2 * (t - 1) * alpha + 2 * (t - 1) / t * message / bandwidth
        ts.append(t)
        compute.append(flops / peak_flops)
        comm.append(2 * ring)  # two all-reduces per layer, forward
    return ts, compute, comm


def plot(measured: dict, out_path: Path, n_layer: int, hidden: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    from common import cluster_specs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # --- left: measured against predicted volume ---------------------------
    if measured:
        sizes = sorted(measured)
        got = [measured[t] / 1e6 for t in sizes]
        want = [predicted_bytes(n_layer, hidden, BATCH, SEQ) / 1e6 for _ in sizes]
        ax1.plot(sizes, got, "o-", color="#2b6cb0", label="measured")
        ax1.plot(sizes, want, "--", color="#dd6b20", label="predicted 4·L·b·s·h")
        ax1.set_ylim(0, max(max(got), max(want)) * 1.4)
    else:
        ax1.text(
            0.5,
            0.5,
            "no measurements\n(is TensorParallelGPT implemented?)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    ax1.set_xlabel("tensor-parallel degree t")
    ax1.set_ylabel("collective payload per rank per step (MB)")
    ax1.set_title("Logical payload stays flat with t")
    ax1.grid(alpha=0.25, ls=":")
    ax1.legend(fontsize=8)

    # --- right: the ceiling, derived ---------------------------------------
    for name, spec in cluster_specs.PROFILES.items():
        # Local transport profiles are useful for communication fitting, but
        # without an accelerator peak they cannot define this ratio.
        if spec["peak_flops_fp16"] <= 0 or spec["interconnect_bytes_per_s"] <= 0:
            continue
        ts, compute, comm = ceiling_curve(
            hidden=4096,
            tokens=8192,
            peak_flops=spec["peak_flops_fp16"],
            bandwidth=spec["interconnect_bytes_per_s"],
            alpha=spec["alpha_s"],
        )
        ratio = [c / max(k, 1e-12) for c, k in zip(compute, comm)]
        ax2.loglog(ts, ratio, "o-", label=name, ms=4)

    ax2.axhline(1.0, ls="--", color="#718096", lw=1)
    ax2.text(1.2, 1.15, "compute = communication", fontsize=8, color="#718096")
    ax2.set_xticks(ts)
    ax2.set_xticklabels([str(t) for t in ts])
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("tensor-parallel degree t")
    ax2.set_ylabel("compute time / communication time")
    ax2.set_title("The scaling ceiling (h=4096, 8192 tokens, fp16)")
    ax2.grid(True, which="both", alpha=0.25, ls=":")
    ax2.legend(fontsize=8)

    fig.suptitle("Lab 2: why tensor parallelism stays inside a fast interconnect domain")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args(argv)

    sizes = [args.nproc] if args.nproc else TP_SIZES
    measured: dict[int, int] = {}

    print("Lab 2 tensor-parallel scaling\n")
    for t in sizes:
        try:
            results = dist_utils.run_distributed(
                _worker,
                world_size=t,
                backend="gloo",
                args=(args.layers, args.hidden, BATCH, SEQ),
                timeout_s=400,
            )
        except NotImplementedError:
            print("  TensorParallelGPT is not implemented yet. Finish step 8 first.")
            break

        volume = max(r["bytes"] for r in results.values())
        measured[t] = volume
        predicted = predicted_bytes(args.layers, args.hidden, BATCH, SEQ)
        print(
            f"  t={t}   measured {volume / 1e6:>7.3f} MB   "
            f"predicted {predicted / 1e6:>7.3f} MB   "
            f"ratio {volume / predicted:.2f}"
        )
        print(f"        {results[0]['counts']}")

    out_dir = timers.runs_dir("lab2_tensor_parallel")
    image = plot(measured, out_dir / "tp_scaling.png", args.layers, args.hidden)

    summary = {
        "allreduces_per_layer_per_step": {"value": 4, "units": "collectives"},
    }
    if 2 in measured:
        summary["measured_payload_l4_b4_s64_h256_tp2"] = {
            "value": measured[2],
            "units": "bytes",
        }
    if 4 in measured:
        summary["measured_payload_l4_b4_s64_h256_tp4"] = {
            "value": measured[4],
            "units": "bytes",
        }

    (out_dir / "measurements.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=True), encoding="utf-8"
    )

    print(f"\n  plot         -> {image}")
    print(f"  measurements -> {out_dir / 'measurements.yaml'}")
    print("\n  The right panel is derived, not measured. Read the crossing points off")
    print("  it and put them in your writeup; that number is what you would take to")
    print("  a capacity discussion.")
    print("\n  next: python tasks.py reconcile LAB=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
