from __future__ import annotations

import numpy as np

__all__ = ["ring_allreduce_numpy", "chunk_boundaries", "reduce_scatter_numpy", "all_gather_numpy"]


def chunk_boundaries(length: int, n_ranks: int) -> list[tuple[int, int]]:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be positive, got {n_ranks}")

    base, remainder = divmod(length, n_ranks)
    bounds = []
    for rank in range(n_ranks):
        start = rank * base + min(rank, remainder)
        stop = start + base + (1 if rank < remainder else 0)
        bounds.append((start, stop))
    return bounds


def _check_buffers(buffers: list[np.ndarray]) -> tuple[int, int, np.dtype]:
    if not buffers:
        raise ValueError("at least one rank is required")
    if any(buffer.ndim != 1 for buffer in buffers):
        raise ValueError("ring buffers must be one-dimensional")

    length = len(buffers[0])
    dtype = buffers[0].dtype
    for buffer in buffers[1:]:
        if len(buffer) != length:
            raise ValueError(f"buffer lengths differ: {length} and {len(buffer)}")
        if buffer.dtype != dtype:
            raise ValueError(f"buffer dtypes differ: {dtype} and {buffer.dtype}")
    return len(buffers), length, dtype


def reduce_scatter_numpy(buffers: list[np.ndarray]) -> list[np.ndarray]:
    n_ranks, length, _ = _check_buffers(buffers)
    bounds = chunk_boundaries(length, n_ranks)
    parts = [[buffer[start:stop].copy() for start, stop in bounds] for buffer in buffers]

    # Snapshot every send before anyone accumulates, or later ranks in the loop
    # receive chunks that already include this round's addition.
    for step in range(n_ranks - 1):
        sends = [parts[rank][(rank - step - 1) % n_ranks].copy() for rank in range(n_ranks)]
        for rank in range(n_ranks):
            recv_idx = (rank - step - 2) % n_ranks
            parts[rank][recv_idx] += sends[(rank - 1) % n_ranks]

    return [parts[rank][rank].copy() for rank in range(n_ranks)]


def all_gather_numpy(chunks: list[np.ndarray], length: int) -> list[np.ndarray]:
    if not chunks:
        raise ValueError("at least one rank is required")
    n_ranks = len(chunks)
    bounds = chunk_boundaries(length, n_ranks)
    dtype = chunks[0].dtype
    for rank, (chunk, (start, stop)) in enumerate(zip(chunks, bounds)):
        expected = stop - start
        if len(chunk) != expected:
            raise ValueError(f"rank {rank} chunk length is {len(chunk)}, expected {expected}")
        if chunk.dtype != dtype:
            raise ValueError(f"chunk dtypes differ: {dtype} and {chunk.dtype}")

    held: list[list[np.ndarray | None]] = [[None for _ in range(n_ranks)] for _ in range(n_ranks)]
    for rank, chunk in enumerate(chunks):
        held[rank][rank] = chunk.copy()

    for step in range(n_ranks - 1):
        send_idx = [(rank - step) % n_ranks for rank in range(n_ranks)]
        sends = [held[rank][send_idx[rank]].copy() for rank in range(n_ranks)]
        for rank in range(n_ranks):
            recv_idx = (rank - step - 1) % n_ranks
            held[rank][recv_idx] = sends[(rank - 1) % n_ranks]

    return [
        np.concatenate([part for part in rank_parts if part is not None]) for rank_parts in held
    ]


def ring_allreduce_numpy(buffers: list[np.ndarray]) -> list[np.ndarray]:
    n_ranks, length, _ = _check_buffers(buffers)
    if n_ranks == 1:
        return [buffers[0].copy()]
    chunks = reduce_scatter_numpy(buffers)
    return all_gather_numpy(chunks, length)
