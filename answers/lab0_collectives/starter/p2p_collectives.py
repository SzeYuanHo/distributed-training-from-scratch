from __future__ import annotations

import torch

from common import collectives, dist_utils

from .ring_numpy import chunk_boundaries

__all__ = [
    "broadcast_linear",
    "broadcast_tree",
    "reduce_to_root",
    "reduce_scatter_ring",
    "all_gather_ring",
    "all_to_all_p2p",
]


def broadcast_linear(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    if not 0 <= src < world:
        raise ValueError(f"source rank {src} is outside world size {world}")
    if rank == src:
        for peer in range(world):
            if peer != src:
                collectives.send(tensor, dst=peer)
    else:
        collectives.recv(tensor, src=src)
    return tensor


def broadcast_tree(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    if not 0 <= src < world:
        raise ValueError(f"source rank {src} is outside world size {world}")

    virtual = (rank - src) % world
    span = 1
    while span < world:
        if virtual < span and virtual + span < world:
            peer = (virtual + span + src) % world
            collectives.send(tensor, dst=peer)
        elif span <= virtual < 2 * span:
            peer = (virtual - span + src) % world
            collectives.recv(tensor, src=peer)
        span *= 2
    return tensor


def reduce_to_root(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    if not 0 <= dst < world:
        raise ValueError(f"destination rank {dst} is outside world size {world}")
    if rank == dst:
        inbox = torch.empty_like(tensor)
        for peer in range(world):
            if peer != dst:
                collectives.recv(inbox, src=peer)
                tensor.add_(inbox)
    else:
        collectives.send(tensor, dst=dst)
    return tensor


def _exchange(send: torch.Tensor, recv: torch.Tensor, dst: int, src: int) -> None:
    # post both sides so a large symmetric exchange cannot deadlock
    recv_work = collectives.irecv(recv, src=src)
    send_work = collectives.isend(send, dst=dst)
    recv_work.wait()
    send_work.wait()


def reduce_scatter_ring(tensor: torch.Tensor) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    flat = tensor.reshape(-1)
    bounds = chunk_boundaries(flat.numel(), world)
    parts = [flat[start:stop].clone() for start, stop in bounds]
    if world == 1:
        return parts[0]

    left, right = (rank - 1) % world, (rank + 1) % world
    for step in range(world - 1):
        send_idx = (rank - step - 1) % world
        recv_idx = (rank - step - 2) % world
        start, stop = bounds[recv_idx]
        inbox = torch.empty(stop - start, dtype=flat.dtype, device=flat.device)
        _exchange(parts[send_idx], inbox, right, left)
        parts[recv_idx].add_(inbox)
    return parts[rank]


def all_gather_ring(chunk: torch.Tensor, full_length: int) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    bounds = chunk_boundaries(full_length, world)
    expected = bounds[rank][1] - bounds[rank][0]
    flat_chunk = chunk.reshape(-1)
    if flat_chunk.numel() != expected:
        raise ValueError(f"rank {rank} chunk has length {flat_chunk.numel()}, expected {expected}")

    held: list[torch.Tensor | None] = [None] * world
    held[rank] = flat_chunk.clone()
    if world == 1:
        return held[0]

    left, right = (rank - 1) % world, (rank + 1) % world
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        start, stop = bounds[recv_idx]
        inbox = torch.empty(stop - start, dtype=flat_chunk.dtype, device=flat_chunk.device)
        send = held[send_idx]
        if send is None:
            raise RuntimeError(f"rank {rank} does not yet hold chunk {send_idx}")
        _exchange(send, inbox, right, left)
        held[recv_idx] = inbox
    return torch.cat([part for part in held if part is not None])


def all_to_all_p2p(tensor: torch.Tensor) -> torch.Tensor:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    flat = tensor.reshape(-1)
    if flat.numel() % world:
        raise ValueError(f"tensor length {flat.numel()} is not divisible by world size {world}")

    width = flat.numel() // world
    inputs = list(flat.split(width))
    outputs = [torch.empty_like(inputs[0]) for _ in range(world)]
    outputs[rank].copy_(inputs[rank])
    for offset in range(1, world):
        dst = (rank + offset) % world
        src = (rank - offset) % world
        _exchange(inputs[dst], outputs[src], dst, src)
    return torch.cat(outputs)
