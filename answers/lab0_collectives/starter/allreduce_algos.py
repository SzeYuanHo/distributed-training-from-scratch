from __future__ import annotations

import math

import torch

from common import collectives, dist_utils

from .p2p_collectives import (
    all_gather_ring,
    broadcast_linear,
    reduce_scatter_ring,
    reduce_to_root,
)

__all__ = ["all_reduce_naive", "all_reduce_ring", "all_reduce_recursive_doubling"]


def all_reduce_naive(tensor: torch.Tensor) -> torch.Tensor:
    reduce_to_root(tensor, dst=0)
    return broadcast_linear(tensor, src=0)


def all_reduce_ring(tensor: torch.Tensor) -> torch.Tensor:
    reduced = reduce_scatter_ring(tensor)
    gathered = all_gather_ring(reduced, tensor.numel())
    tensor.copy_(gathered.reshape_as(tensor))
    return tensor


def all_reduce_recursive_doubling(tensor: torch.Tensor) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    if world < 1 or world & (world - 1):
        raise ValueError(
            f"recursive doubling requires a power-of-two world size; got {world}; use the ring instead"
        )

    # Whole-buffer variant: log2(N) rounds exchanging everything, so log2(N)*M
    # bytes against the ring's 2(N-1)/N*M. Level at N=2, and the ring is ahead on
    # volume for every N above that (3M vs 1.75M at N=8, 6M vs 1.97M at N=64)
    # while this one keeps the latency win. Only worth it on small messages.
    for bit in range(int(math.log2(world))):
        peer = rank ^ (1 << bit)
        inbox = torch.empty_like(tensor)
        recv_work = collectives.irecv(inbox, src=peer)
        send_work = collectives.isend(tensor, dst=peer)
        recv_work.wait()
        send_work.wait()
        tensor.add_(inbox)
    return tensor
