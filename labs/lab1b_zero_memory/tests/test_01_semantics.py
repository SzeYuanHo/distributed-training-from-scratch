"""Tier 1: does ZeRO-1 train the same model as DDP, on less memory, for the same bytes?

Three claims, three tests:

1. Same loss curve as plain DDP. Sharding the optimizer must not change the maths.
2. Optimizer state falls as 1/N. That is the entire point.
3. Total communication equals DDP's. This is the one people do not believe.
"""

from __future__ import annotations

import pytest
import torch

from common import comm_counter, dist_utils, reference_gpt, testing
from labs.lab1b_zero_memory.reference import ddp_baseline
from labs.lab1b_zero_memory.starter import zero1

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(300)]

STEPS = 10
BATCH = 4
BLOCK = 64
LR = 1e-3
BETAS = (0.9, 0.95)


def _build_model():
    from common import data as data_mod

    torch.manual_seed(1337)
    dataset = data_mod.load_dataset()
    return reference_gpt.make_model("tiny", block_size=BLOCK, vocab_size=dataset.vocab_size)


# --------------------------------------------------------------------------- #
# sharding
# --------------------------------------------------------------------------- #


def test_shard_parameters_covers_everything_once():
    params = [torch.nn.Parameter(torch.randn(n)) for n in (1000, 500, 500, 10, 10, 10)]
    shards = zero1.shard_parameters(params, world_size=3)

    assert len(shards) == 3
    flat = [p for shard in shards for p in shard]
    assert len(flat) == len(params), f"expected {len(params)} parameters, saw {len(flat)}"
    assert {id(p) for p in flat} == {id(p) for p in params}


def test_shard_parameters_balances_bytes():
    """One dominant tensor plus many small ones is the realistic case.

    Round-robin gives one rank the big tensor and everyone else the scraps, and
    the memory saving mostly evaporates. Greedy largest-first is a few lines and
    much better.
    """
    params = [torch.nn.Parameter(torch.randn(10_000))] + [
        torch.nn.Parameter(torch.randn(100)) for _ in range(20)
    ]
    shards = zero1.shard_parameters(params, world_size=4)

    totals = [sum(p.numel() for p in shard) for shard in shards]
    assert all(t > 0 for t in totals), (
        f"a rank got an empty shard: {totals}. That rank has no optimizer state and "
        "will produce a shape mismatch at all-gather time."
    )

    imbalance = max(totals) / max(min(totals), 1)
    largest = max(p.numel() for p in params)
    remaining_per_rank = (sum(p.numel() for p in params) - largest) / (len(shards) - 1)
    unavoidable = largest / remaining_per_rank
    assert imbalance <= unavoidable * 1.2, (
        f"shard sizes are {totals}, an imbalance of {imbalance:.1f}x. With one tensor "
        f"holding 83% of the elements, even the continuous lower bound is "
        f"{unavoidable:.1f}x. The result should stay within 20% of that bound; "
        "round-robin does not."
    )


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #


def _w_ddp_baseline():
    model = _build_model()
    wrapper = ddp_baseline.PlainDDP(model)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=LR, weight_decay=0.0, betas=BETAS)

    with comm_counter.CommCounter() as counter:
        losses = ddp_baseline.train_steps(wrapper, optimizer, STEPS, batch=BATCH, block=BLOCK)

    return {
        "rank": dist_utils.rank(),
        "losses": losses,
        "bytes": counter.total_bytes,
        "counts": counter.counts(),
        "wire_bytes": _estimated_wire_bytes(counter.records),
        "state_bytes": ddp_baseline.optimizer_state_bytes(optimizer),
    }


def _w_zero1():
    model = _build_model()
    dist_utils.sync_module_params(model)
    optimizer = zero1.ZeRO1Optimizer(model.parameters(), lr=LR, weight_decay=0.0, betas=BETAS)

    with comm_counter.CommCounter() as counter:
        losses = ddp_baseline.train_steps(model, optimizer, STEPS, batch=BATCH, block=BLOCK)

    for name, param in model.named_parameters():
        testing.assert_close_across_ranks(
            param.data, rtol=1e-5, atol=1e-7, name=f"post-step {name}"
        )

    return {
        "rank": dist_utils.rank(),
        "losses": losses,
        "bytes": counter.total_bytes,
        "counts": counter.counts(),
        "wire_bytes": _estimated_wire_bytes(counter.records),
        "state_bytes": optimizer.local_state_bytes(),
    }


def _estimated_wire_bytes(records: list[comm_counter.CommRecord]) -> float:
    """Convert recorded buffer sizes to per-rank algorithmic wire volume."""
    total = 0.0
    for record in records:
        world = record.group_size
        if record.op == "all_reduce":
            total += 2 * (world - 1) / world * record.n_bytes
        elif record.op == "reduce_scatter":
            total += (world - 1) / world * record.n_bytes
        elif record.op == "all_gather":
            total += (world - 1) * record.n_bytes
    return total


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.slow
def test_zero1_loss_matches_ddp(world):
    """Sharding where the optimizer state lives must not change the maths."""
    baseline = dist_utils.run_distributed(
        _w_ddp_baseline, world_size=world, backend="gloo", timeout_s=280
    )
    sharded = dist_utils.run_distributed(_w_zero1, world_size=world, backend="gloo", timeout_s=280)

    for rank in range(world):
        testing.assert_curves_close(
            sharded[rank]["losses"],
            baseline[rank]["losses"],
            rtol=3e-2,
            name=f"ZeRO-1 vs DDP loss curve (rank {rank}, world {world})",
        )


@pytest.mark.parametrize("world", [2, 4])
def test_zero1_keeps_parameters_identical_across_ranks(world):
    """The all-gather has to run, and it has to run after the local step.

    Skip it, or do it before stepping, and each rank ends up holding a different
    model while the loss keeps falling. Same failure as Lab H hazard 2, reached
    by a different route.
    """
    dist_utils.run_distributed(_w_zero1, world_size=world, backend="gloo", timeout_s=280)


# --------------------------------------------------------------------------- #
# the memory claim
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world", [2, 4])
def test_optimizer_state_shrinks_with_world_size(world):
    """State per rank should be roughly 1/N of DDP's. This is the whole point.

    The bug this catches: building AdamW over every parameter and only stepping
    some of them. Correct answer, no saving, and Adam allocates m and v for
    everything on the first step regardless of which ones you touched.
    """
    baseline = dist_utils.run_distributed(
        _w_ddp_baseline, world_size=world, backend="gloo", timeout_s=280
    )
    sharded = dist_utils.run_distributed(_w_zero1, world_size=world, backend="gloo", timeout_s=280)

    full = baseline[0]["state_bytes"]
    per_rank = max(r["state_bytes"] for r in sharded.values())

    assert full > 0, (
        "the DDP baseline reported zero optimizer state, which means it was measured "
        "before the first step(). Adam allocates m and v lazily."
    )

    ratio = per_rank / full
    assert ratio < (1.0 / world) * 1.6, (
        f"at N={world}, ZeRO-1 holds {per_rank:,} bytes of optimizer state per rank "
        f"against DDP's {full:,}, a ratio of {ratio:.2f}. Expected roughly "
        f"{1 / world:.2f}.\n"
        "  A ratio near 1.0 means the inner AdamW was built over every parameter "
        "rather than over this rank's shard."
    )


# --------------------------------------------------------------------------- #
# the communication claim
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world", [2, 4])
def test_zero1_communication_equals_ddp(world):
    """The result that surprises people, asserted.

    An all-reduce *is* a reduce-scatter followed by an all-gather. Replacing one
    with the other two changes where the bytes go, not how many there are. So
    ZeRO-1 buys you memory for free in bandwidth terms.

    The counter records API buffer sizes, not wire volume: a reduce-scatter sees
    its full concatenated input while an all-gather sees one rank's input. Convert
    those records using each collective's algorithmic volume before comparing.
    Padding whole-parameter shards can still add a small overhead.
    """
    baseline = dist_utils.run_distributed(
        _w_ddp_baseline, world_size=world, backend="gloo", timeout_s=280
    )
    sharded = dist_utils.run_distributed(_w_zero1, world_size=world, backend="gloo", timeout_s=280)

    ddp_buffer_bytes = baseline[0]["bytes"]
    zero_buffer_bytes = sharded[0]["bytes"]
    ddp_bytes = baseline[0]["wire_bytes"]
    zero_bytes = sharded[0]["wire_bytes"]
    ratio = zero_bytes / ddp_bytes if ddp_bytes else float("inf")

    print(
        f"\n  world {world}: estimated DDP wire volume {ddp_bytes:,.0f} bytes, "
        f"ZeRO-1 {zero_bytes:,.0f} ({ratio:.2f}x)"
    )
    print(f"    raw API buffers: DDP {ddp_buffer_bytes:,}, ZeRO-1 {zero_buffer_bytes:,}")
    print(f"    DDP    {baseline[0]['counts']}")
    print(f"    ZeRO-1 {sharded[0]['counts']}")

    assert 0.85 < ratio < 1.2, (
        f"ZeRO-1 moved {ratio:.2f}x the bytes DDP did at N={world}.\n"
        "  These should be about equal: reduce-scatter plus all-gather is exactly "
        "what an all-reduce decomposes into.\n"
        "  A ratio near 2 usually means an all-reduce is still in there where a "
        "reduce-scatter belongs."
    )
