"""Tier 2: the only tier that can say anything honest about speed.

Skipped automatically with fewer than two CUDA devices. When Kaggle currently
offers its "GPU T4 x 2" accelerator, the optional
``notebooks/kaggle_2xT4_tier2.ipynb`` shows one way to run them.

Why nothing here runs at Tier 1: on one machine over loopback, communication
dominates and your from-scratch ring will measure slower than a single process.
That is not a bug you introduced, it is what the cost model predicts, and
presenting a gloo timing as evidence of a distributed speedup would be dishonest.
Tier 1 tells you what happens; Tier 2 tells you how fast.
"""

from __future__ import annotations

import pytest
import torch

from common import dist_utils, testing
from labs.lab0_collectives.starter import allreduce_algos

pytestmark = [pytest.mark.tier2, pytest.mark.timeout(600)]


def _time_op(fn, tensor: torch.Tensor, iters: int = 20, warmup: int = 5) -> float:
    """Median seconds per call, CUDA-synchronised. Median, not mean: one scheduler
    hiccup skews a mean and the median shrugs it off."""
    import time

    for _ in range(warmup):
        fn(tensor.clone())
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        work = tensor.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(work)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)

    samples.sort()
    return samples[len(samples) // 2]


def _w_compare_algorithms(n_bytes: int):
    rank = dist_utils.rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    n_elem = n_bytes // 4
    tensor = torch.randn(n_elem, device=device) + rank

    return {
        "rank": rank,
        "ring": _time_op(allreduce_algos.all_reduce_ring, tensor),
        "naive": _time_op(allreduce_algos.all_reduce_naive, tensor),
        "torch": _time_op(torch.distributed.all_reduce, tensor),
    }


@testing.requires_cuda_devices(2)
@pytest.mark.parametrize("n_bytes", [4 * 1024 * 1024, 32 * 1024 * 1024])
def test_ring_beats_naive_on_large_messages(n_bytes):
    """Above a few megabytes the root's link is the bottleneck and the ring wins.

    This is the one claim in the lab that genuinely needs real hardware. On
    loopback both algorithms are dominated by python overhead and the comparison
    says nothing.
    """
    results = dist_utils.run_distributed(
        _w_compare_algorithms, world_size=2, backend="nccl", args=(n_bytes,)
    )

    ring = max(r["ring"] for r in results.values())
    naive = max(r["naive"] for r in results.values())

    assert ring < naive, (
        f"at {n_bytes / 1e6:.0f} MB the ring took {ring * 1e3:.2f} ms and the naive "
        f"gather-to-root took {naive * 1e3:.2f} ms. The ring should win here: the "
        "naive version funnels (N-1)M bytes through one link."
    )


@testing.requires_cuda_devices(2)
def test_report_gap_to_nccl():
    """Not a pass/fail claim. Prints how far your ring is from the real thing.

    Expect a factor of several. NCCL pipelines segments so the reduce-scatter of
    segment k overlaps the all-gather of segment k-1, switches protocol by size,
    and issues no python per message. Your writeup should account for the gap
    rather than apologise for it.
    """
    results = dist_utils.run_distributed(
        _w_compare_algorithms, world_size=2, backend="nccl", args=(16 * 1024 * 1024,)
    )

    ring = max(r["ring"] for r in results.values())
    native = max(r["torch"] for r in results.values())

    print("\n  16 MB all-reduce on 2 devices:")
    print(f"    your ring        {ring * 1e3:8.3f} ms")
    print(f"    dist.all_reduce  {native * 1e3:8.3f} ms")
    print(f"    ratio            {ring / native:8.2f}x")
    print("    Where the gap goes: segment pipelining, protocol switching by size,")
    print("    and zero python per message. Put your number in predictions.yaml as")
    print("    ring_vs_torch_slowdown_factor and see how close you guessed.")

    assert ring > 0 and native > 0
