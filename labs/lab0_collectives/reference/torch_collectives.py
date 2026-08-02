"""Ground truth: torch's own collectives, wrapped so the tests read cleanly.

Nothing clever here. These call ``torch.distributed`` directly and exist so the
tests can say ``assert_close(mine, reference.all_reduce(x))`` instead of
inlining the setup every time.

Deliberately bypasses ``common.collectives``, so reference calls do not pollute
the CommCounter assertions your implementation is being judged on.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = [
    "all_reduce",
    "broadcast",
    "reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "expected_sum",
]


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    out = tensor.clone()
    dist.all_reduce(out, op=op)
    return out


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    out = tensor.clone()
    dist.broadcast(out, src=src)
    return out


def reduce(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    out = tensor.clone()
    dist.reduce(out, dst=dst)
    return out


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size()
    buckets = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(buckets, tensor)
    return torch.cat(buckets)


def reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter over a buffer that divides evenly by the world size."""
    world = dist.get_world_size()
    chunks = list(tensor.chunk(world))
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, [c.contiguous() for c in chunks])
    return out


def all_to_all(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(tensor)
    dist.all_to_all_single(out, tensor)
    return out


def expected_sum(make_tensor, world_size: int) -> torch.Tensor:
    """What an all-reduce should produce, computed without any communication.

    ``make_tensor(rank)`` builds the contribution rank ``r`` would have made. The
    tests use this so the expected value does not itself depend on a collective
    working, which would make a broken collective agree with itself.
    """
    total = make_tensor(0).clone()
    for r in range(1, world_size):
        total += make_tensor(r)
    return total
