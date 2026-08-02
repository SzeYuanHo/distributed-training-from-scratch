"""Instrumented wrappers around ``torch.distributed``.

**Lab code calls these, not ``torch.distributed`` directly.** Every wrapper
records the op and its byte count into whatever ``CommCounter`` is active, then
delegates. That is what makes "assert this MLP block issues exactly one
all-reduce" a test you can write.

The signatures match ``torch.distributed`` so switching a line back and forth
costs nothing. Where behaviour differs, the docstring says so.

Async handles are wrapped too. ``AsyncHandle.wait()`` records a separate
``wait`` event, which is how Lab 1 can tell the difference between "the
all-reduce was issued" and "the all-reduce finished", and therefore how the
"read ``.grad`` before waiting" bug becomes visible instead of silent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from common import comm_counter as _cc

__all__ = [
    "ReduceOp",
    "AsyncHandle",
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "broadcast",
    "reduce",
    "all_to_all",
    "all_to_all_single",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "barrier",
    "P2POp",
]

P2POp = dist.P2POp


def _nbytes(t: Any) -> int:
    return _cc.tensor_bytes(t)


def _group_size(group) -> int:
    if group is None:
        return dist.get_world_size() if dist.is_initialized() else 1
    return dist.get_world_size(group=group)


class AsyncHandle:
    """Wraps a ``dist.Work`` so the wait shows up in the counter too.

    Why bother: an async collective that is never waited on produces a tensor
    that is *partially* reduced. It has the right shape, the right dtype, and
    plausible values, and your loss still goes down. Recording waits separately
    means a test can assert "every all-reduce issued this step was also waited
    on", which turns a silent bug into a loud one.
    """

    __slots__ = ("_work", "_op", "_bytes", "_waited")

    def __init__(self, work, op: str, n_bytes: int):
        self._work = work
        self._op = op
        self._bytes = n_bytes
        self._waited = False

    def wait(self, timeout=None) -> None:
        if self._work is not None:
            self._work.wait() if timeout is None else self._work.wait(timeout)
        self._waited = True
        _cc.record(f"wait[{self._op}]", 0, 1, waited_bytes=self._bytes)

    def is_completed(self) -> bool:
        return self._work is None or self._work.is_completed()

    @property
    def waited(self) -> bool:
        return self._waited

    def __repr__(self) -> str:
        return f"<AsyncHandle {self._op} {self._bytes}B waited={self._waited}>"


def _maybe_async(work, async_op: bool, op: str, n_bytes: int):
    if not async_op:
        return None
    return AsyncHandle(work, op, n_bytes)


# --------------------------------------------------------------------------- #
# reductions
# --------------------------------------------------------------------------- #


def all_reduce(tensor: torch.Tensor, op=ReduceOp.SUM, group=None, async_op: bool = False):
    """Every rank ends up with the reduction of every rank's input, in place.

    Bytes recorded are the *buffer* size, not the wire volume. A ring
    all-reduce actually moves ``2(N-1)/N`` times that. Lab 0 asks you to derive
    the difference, and the counter deliberately reports the simpler number so
    the derivation stays yours.
    """
    n = _nbytes(tensor)
    _cc.record("all_reduce", n, _group_size(group), reduce_op=str(op))
    work = dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "all_reduce", n)


def reduce(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM, group=None, async_op: bool = False):
    """Reduce onto one rank. Only ``dst`` holds a meaningful result afterwards."""
    n = _nbytes(tensor)
    _cc.record("reduce", n, _group_size(group), dst=dst, reduce_op=str(op))
    work = dist.reduce(tensor, dst=dst, op=op, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "reduce", n)


def reduce_scatter(
    output: torch.Tensor,
    input_list: Sequence[torch.Tensor],
    op=ReduceOp.SUM,
    group=None,
    async_op: bool = False,
):
    """Reduce across ranks, then scatter so each rank keeps one slice."""
    n = _nbytes(list(input_list))
    _cc.record("reduce_scatter", n, _group_size(group), reduce_op=str(op))
    work = dist.reduce_scatter(output, list(input_list), op=op, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "reduce_scatter", n)


def reduce_scatter_tensor(
    output: torch.Tensor, input: torch.Tensor, op=ReduceOp.SUM, group=None, async_op: bool = False
):
    """Flat-buffer reduce-scatter. ``input`` must be ``world_size`` times ``output``."""
    n = _nbytes(input)
    _cc.record("reduce_scatter", n, _group_size(group), reduce_op=str(op), flat=True)
    implementation = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    work = implementation(output, input, op=op, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "reduce_scatter", n)


# --------------------------------------------------------------------------- #
# data movement
# --------------------------------------------------------------------------- #


def all_gather(
    tensor_list: Sequence[torch.Tensor], tensor: torch.Tensor, group=None, async_op: bool = False
):
    """Each rank contributes ``tensor``; every rank receives all of them."""
    n = _nbytes(tensor)
    _cc.record("all_gather", n, _group_size(group))
    work = dist.all_gather(list(tensor_list), tensor, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "all_gather", n)


def all_gather_into_tensor(
    output: torch.Tensor, input: torch.Tensor, group=None, async_op: bool = False
):
    """Flat-buffer all-gather. ``output`` must be ``world_size`` times ``input``."""
    n = _nbytes(input)
    _cc.record("all_gather", n, _group_size(group), flat=True)
    implementation = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    work = implementation(output, input, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "all_gather", n)


def broadcast(tensor: torch.Tensor, src: int, group=None, async_op: bool = False):
    """One rank's tensor overwrites everyone else's."""
    n = _nbytes(tensor)
    _cc.record("broadcast", n, _group_size(group), src=src)
    work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    return _maybe_async(work, async_op, "broadcast", n)


def all_to_all(
    output_tensor_list: Sequence[torch.Tensor],
    input_tensor_list: Sequence[torch.Tensor],
    group=None,
    async_op: bool = False,
):
    """Rank i's j-th chunk goes to rank j's i-th slot. A distributed transpose."""
    n = _nbytes(list(input_tensor_list))
    _cc.record("all_to_all", n, _group_size(group))
    work = dist.all_to_all(
        list(output_tensor_list), list(input_tensor_list), group=group, async_op=async_op
    )
    return _maybe_async(work, async_op, "all_to_all", n)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op: bool = False,
):
    n = _nbytes(input)
    _cc.record("all_to_all", n, _group_size(group), flat=True)
    work = dist.all_to_all_single(
        output, input, output_split_sizes, input_split_sizes, group=group, async_op=async_op
    )
    return _maybe_async(work, async_op, "all_to_all", n)


# --------------------------------------------------------------------------- #
# point to point
# --------------------------------------------------------------------------- #


def send(tensor: torch.Tensor, dst: int, group=None, tag: int = 0) -> None:
    """Blocking send. Pairs with a ``recv`` on ``dst``.

    Blocking is the operative word. If peers all call ``send`` before any posts
    ``recv``, progress may stop; transport buffering can hide the defect for
    some payloads but is not a correctness guarantee. Lab 0 demonstrates that
    hazard and Lab 3 avoids it.
    """
    n = _nbytes(tensor)
    _cc.record("send", n, 2, dst=dst, tag=tag)
    dist.send(tensor, dst=dst, group=group, tag=tag)


def recv(tensor: torch.Tensor, src: int | None = None, group=None, tag: int = 0) -> int:
    """Blocking receive into ``tensor``. Returns the source rank.

    ``tensor`` must already have the right shape and dtype. There is no
    negotiation: a receiver that guesses wrong either hangs or silently
    reinterprets the bytes. Lab 3 handles this by sending a metadata tensor
    first.
    """
    n = _nbytes(tensor)
    _cc.record("recv", n, 2, src=src, tag=tag)
    return dist.recv(tensor, src=src, group=group, tag=tag)


def isend(tensor: torch.Tensor, dst: int, group=None, tag: int = 0) -> AsyncHandle:
    n = _nbytes(tensor)
    _cc.record("isend", n, 2, dst=dst, tag=tag)
    return AsyncHandle(dist.isend(tensor, dst=dst, group=group, tag=tag), "isend", n)


def irecv(tensor: torch.Tensor, src: int | None = None, group=None, tag: int = 0) -> AsyncHandle:
    n = _nbytes(tensor)
    _cc.record("irecv", n, 2, src=src, tag=tag)
    return AsyncHandle(dist.irecv(tensor, src=src, group=group, tag=tag), "irecv", n)


def batch_isend_irecv(p2p_ops: Sequence[dist.P2POp]) -> list:
    """Issue a batch of point-to-point ops together.

    This is the recommended way to exchange activations in a pipeline. The
    backend gets to see every send and receive at once, so the symmetric
    "everyone sends first" deadlock cannot happen. Build the op list with
    ``collectives.P2POp(dist.isend, tensor, peer)``.
    """
    for op in p2p_ops:
        name = "isend" if op.op is dist.isend else "irecv"
        _cc.record(name, _nbytes(op.tensor), 2, peer=op.peer, batched=True)
    return dist.batch_isend_irecv(list(p2p_ops))


def barrier(group=None, async_op: bool = False):
    """Synchronise. Records zero bytes but still counts as a call."""
    _cc.record("barrier", 0, _group_size(group))
    work = dist.barrier(group=group, async_op=async_op)
    return _maybe_async(work, async_op, "barrier", 0)
