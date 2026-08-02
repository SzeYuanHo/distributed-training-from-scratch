"""Tier 2: the only place a speedup claim is honest.

Skipped without two CUDA devices. The optional Kaggle notebook can run this when a
compatible two-GPU accelerator is available.

Tier 1 cannot answer these questions. Four gloo ranks on one laptop share a memory
bus and a core count, so DDP there adds communication without adding compute and
measures slower than one process. That is the cost model working, not a bug, and
presenting a gloo timing as a scaling result would be dishonest.
"""

from __future__ import annotations

import time

import pytest
import torch

from common import dist_utils, testing, timers
from labs.lab1_ddp_bucketing.reference import single_process as ref
from labs.lab1_ddp_bucketing.starter import bucketed_ddp

pytestmark = [pytest.mark.tier2, pytest.mark.timeout(900)]


def _w_throughput(bucket_bytes: int, steps: int, overlap: bool):
    from common import data as data_mod

    rank = dist_utils.rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    config = ref.TrainConfig(
        preset="small",
        steps=steps,
        per_rank_batch=16,
        world_size=dist_utils.world_size(),
        block_size=128,
    )
    model = ref.build_model(config).to(device)
    wrapper = bucketed_ddp.BucketedDDP(model, bucket_bytes=bucket_bytes, overlap=overlap)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=config.lr)
    dataset = data_mod.load_dataset()

    log = timers.EventLog()

    for step in range(steps):
        x, y = data_mod.get_batch(
            "train",
            batch_size=config.per_rank_batch,
            block_size=config.block_size,
            rank=rank,
            world_size=config.world_size,
            step=step,
            device=device,
            dataset=dataset,
        )
        if step == 3:  # warmup done
            torch.cuda.synchronize()
            start = time.perf_counter()

        with log.span("forward", "compute"):
            _logits, loss = wrapper(x, y)
        optimizer.zero_grad(set_to_none=True)
        with log.span("backward", "compute"):
            loss.backward()
        with log.span("finalize", "comm"):
            wrapper.finalize()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = (steps - 3) * config.per_rank_batch * config.block_size

    merged = log.gather()
    return {
        "rank": rank,
        "tokens_per_s": tokens / elapsed,
        "occupancy": log.occupancy(merged),
    }


@testing.requires_cuda_devices(2)
def test_scaling_efficiency_on_two_devices():
    """Two devices should beat one. Not by 2x, and the gap is the lab's real subject."""
    single = dist_utils.run_distributed(
        _w_throughput, world_size=1, backend="gloo", args=(1 << 22, 15, True)
    )
    double = dist_utils.run_distributed(
        _w_throughput, world_size=2, backend="nccl", args=(1 << 22, 15, True)
    )

    one = single[0]["tokens_per_s"]
    two = sum(r["tokens_per_s"] for r in double.values())
    efficiency = two / (2 * one)

    print(f"\n  1 device   {one:>10,.0f} tokens/s")
    print(f"  2 devices  {two:>10,.0f} tokens/s")
    print(f"  scaling efficiency  {efficiency:.1%}")
    print("  Put this in predictions.yaml as tier2_scaling_efficiency_2xT4 and")
    print("  reconcile it against what you guessed.")

    assert efficiency > 0.5, (
        f"scaling efficiency {efficiency:.1%} is below 50%, which on 2 T4s with this "
        "model size suggests communication is not overlapping at all. Check that "
        "buckets flush with async_op=True and that finalize() is not being called "
        "inside the backward pass."
    )


@testing.requires_cuda_devices(2)
def test_overlap_beats_no_overlap():
    """The payoff. Turning overlap off should measurably cost you."""
    overlapped = dist_utils.run_distributed(
        _w_throughput, world_size=2, backend="nccl", args=(1 << 22, 15, True)
    )
    serial = dist_utils.run_distributed(
        _w_throughput, world_size=2, backend="nccl", args=(1 << 22, 15, False)
    )

    fast = sum(r["tokens_per_s"] for r in overlapped.values())
    slow = sum(r["tokens_per_s"] for r in serial.values())

    exposed_overlapped = overlapped[0]["occupancy"].get("comm_exposed", 0)
    exposed_serial = serial[0]["occupancy"].get("comm_exposed", 0)

    print(f"\n  overlapped  {fast:>10,.0f} tokens/s   exposed comm {exposed_overlapped:.1%}")
    print(f"  serial      {slow:>10,.0f} tokens/s   exposed comm {exposed_serial:.1%}")

    assert fast >= slow * 0.95, (
        "overlapping communication made things slower, which means the async path is "
        "doing extra work rather than hiding it. Check for a synchronise inside the "
        "hook."
    )
