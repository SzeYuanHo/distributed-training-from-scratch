"""Tier 2: pipeline timings on real hardware.

Skipped without two CUDA devices. On CPU, stage compute is fast and gloo latency
is high, so the measured bubble is dominated by communication rather than by the
schedule. Two T4s give a compute-to-communication ratio where the bubble is
actually the bubble.
"""

from __future__ import annotations

import pytest
import torch

from common import dist_utils, testing, timers
from labs.lab3_pipeline_parallel.reference import single_process as ref
from labs.lab3_pipeline_parallel.starter import partition, schedule_1f1b, schedule_afab

pytestmark = [pytest.mark.tier2, pytest.mark.timeout(900)]


def _w_timed(kind: str, n_micro: int):
    rank = dist_utils.rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    config = ref.PipeConfig(
        n_layer=8, n_embd=512, n_head=8, block_size=128, batch_size=32, n_micro=n_micro
    )
    model = ref.build_model(config).to(device)
    stage = partition.split_model(model, dist_utils.world_size())[rank]

    x, y = ref.get_batch(config)
    x, y = x.to(device), y.to(device)
    micro = config.micro_batch_size
    xs = [x[i * micro : (i + 1) * micro] for i in range(n_micro)]
    ys = [y[i * micro : (i + 1) * micro] for i in range(n_micro)]

    log = timers.EventLog()
    runner = schedule_afab.run_afab if kind == "afab" else schedule_1f1b.run_1f1b

    runner(stage, n_micro, xs, ys)  # warmup
    torch.cuda.synchronize()

    loss = runner(stage, n_micro, xs, ys, event_log=log)
    torch.cuda.synchronize()

    merged = log.gather()
    return {
        "rank": rank,
        "loss": float(loss),
        "occupancy": log.occupancy(merged),
        "peak": torch.cuda.max_memory_allocated(device),
    }


@testing.requires_cuda_devices(2)
@pytest.mark.parametrize("n_micro", [4, 16])
def test_bubble_shrinks_with_more_microbatches(n_micro):
    """Elapsed idle fraction, on hardware where the measurement means something."""
    results = dist_utils.run_distributed(
        _w_timed, world_size=2, backend="nccl", args=("1f1b", n_micro)
    )
    occ = results[0]["occupancy"]
    theory = (2 - 1) / (n_micro + 2 - 1)
    measured = max(0.0, 1.0 - occ.get("compute", 0.0))

    print(f"\n  p=2, m={n_micro}: theory {theory:.3f}, measured {measured:.3f}")
    assert measured < 0.9, "the pipeline spent almost all its time idle; check the schedule"


@testing.requires_cuda_devices(2)
def test_1f1b_uses_less_memory_than_afab_on_device():
    """On CUDA the memory measurement is real, unlike the CPU RSS proxy at Tier 1."""
    afab = dist_utils.run_distributed(_w_timed, world_size=2, backend="nccl", args=("afab", 16))
    one = dist_utils.run_distributed(_w_timed, world_size=2, backend="nccl", args=("1f1b", 16))

    afab_peak = max(r["peak"] for r in afab.values())
    one_peak = max(r["peak"] for r in one.values())

    print(
        f"\n  AFAB {afab_peak / 1e6:.1f} MB   1F1B {one_peak / 1e6:.1f} MB   "
        f"ratio {afab_peak / one_peak:.2f}x"
    )

    assert one_peak < afab_peak, (
        f"1F1B peaked at {one_peak / 1e6:.1f} MB against AFAB's {afab_peak / 1e6:.1f} MB. "
        "On real device memory the difference should be clear."
    )
