from __future__ import annotations

import torch
import torch.distributed as dist

from common import collectives, dist_utils

__all__ = ["send_activation", "recv_activation", "exchange", "send_recv_pairs"]


def send_activation(tensor: torch.Tensor, dst: int, group=None) -> None:
    collectives.send(tensor.contiguous(), dst=dst, group=group)


def recv_activation(
    shape: tuple,
    dtype: torch.dtype,
    src: int,
    group=None,
    requires_grad: bool = True,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=dtype, device=dist_utils.device_for_rank())
    collectives.recv(tensor, src=src, group=group)
    # The graph does not cross the process boundary, so this arrives with no
    # history. Making it a leaf is what gives it a .grad to send back upstream.
    return tensor.requires_grad_(requires_grad)


def exchange(
    send_tensor: torch.Tensor | None,
    send_to: int | None,
    recv_shape: tuple | None,
    recv_dtype: torch.dtype | None,
    recv_from: int | None,
    group=None,
) -> torch.Tensor | None:
    ops = []
    received = None
    if send_to is not None:
        if send_tensor is None:
            raise ValueError("send_tensor is required when send_to is set")
        send_tensor = send_tensor.contiguous()
        ops.append(collectives.P2POp(dist.isend, send_tensor, send_to, group))
    if recv_from is not None:
        if recv_shape is None or recv_dtype is None:
            raise ValueError("receive shape and dtype are required")
        device = send_tensor.device if send_tensor is not None else dist_utils.device_for_rank()
        received = torch.empty(recv_shape, dtype=recv_dtype, device=device)
        ops.append(collectives.P2POp(dist.irecv, received, recv_from, group))
    for request in collectives.batch_isend_irecv(ops):
        request.wait()
    return received


def send_recv_pairs(stage_index: int, n_stages: int) -> tuple[int | None, int | None]:
    previous = stage_index - 1 if stage_index > 0 else None
    following = stage_index + 1 if stage_index + 1 < n_stages else None
    return previous, following
