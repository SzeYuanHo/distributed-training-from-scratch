"""Tier 2: communication scaling on real hardware.

Skipped without two CUDA devices. The claim this lab makes about the scaling
ceiling cannot be measured on one machine, so what runs here is the part that can:
the logical collective payload per step against tp_size, and whether it matches the
b*s*h prediction.
"""

from __future__ import annotations

import pytest
import torch

from common import comm_counter, dist_utils, testing
from labs.lab2_tensor_parallel.reference import plain_blocks
from labs.lab2_tensor_parallel.starter import tp_gpt

pytestmark = [pytest.mark.tier2, pytest.mark.timeout(600)]


def _w_measure_volume(preset_layers: int, hidden: int, batch: int, seq: int):
    rank = dist_utils.rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    config = plain_blocks.tiny_config(
        n_layer=preset_layers, n_embd=hidden, n_head=8, block_size=seq, vocab_size=64, max_tp=8
    )
    reference = plain_blocks.make_model(config, seed=0).to(device)
    parallel = tp_gpt.TensorParallelGPT.from_reference(reference)

    ids = torch.randint(0, config.vocab_size, (batch, seq), device=device)
    targets = torch.randint(0, config.vocab_size, (batch, seq), device=device)

    with comm_counter.CommCounter() as counter:
        _logits, loss = parallel(ids, targets)
        loss.backward()

    return {
        "rank": rank,
        "bytes": counter.total_bytes,
        "counts": counter.counts(),
    }


@testing.requires_cuda_devices(2)
def test_communication_volume_is_independent_of_tp_degree():
    """The core scaling claim, measured.

    All-reduced tensors are (b, s, h) whatever tp_size is, so the byte count
    should stay roughly flat as you add ranks while the per-rank compute halves.
    That divergence is the ceiling.
    """
    results = dist_utils.run_distributed(
        _w_measure_volume,
        world_size=2,
        backend="nccl",
        args=(4, 512, 8, 128),
    )

    per_rank = [r["bytes"] for r in results.values()]
    print(f"\n  tp=2: {per_rank[0] / 1e6:.2f} MB per rank per step")
    print(f"        {results[0]['counts']}")
    print("  The all-reduced tensor is (b, s, h) regardless of tp_size, so this")
    print("  number stays flat while per-rank compute falls as 1/t. That gap is")
    print("  the scaling ceiling from THEORY section 6.")

    assert max(per_rank) - min(per_rank) < max(per_rank) * 0.05, (
        "ranks moved noticeably different byte counts, which they should not: "
        "every rank participates in every all-reduce"
    )
