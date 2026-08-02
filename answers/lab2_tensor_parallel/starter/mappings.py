from __future__ import annotations

import torch
import torch.distributed as dist

from common import collectives, dist_utils

__all__ = [
    "CopyToTensorParallelRegion",
    "ReduceFromTensorParallelRegion",
    "ScatterToTensorParallelRegion",
    "GatherFromTensorParallelRegion",
    "copy_to_tp_region",
    "reduce_from_tp_region",
    "scatter_to_tp_region",
    "gather_from_tp_region",
    "split_along_last_dim",
    "gather_along_last_dim",
]


def _rank(group=None) -> int:
    if group is None:
        return dist_utils.rank()
    return dist.get_rank(group=group)


def _world_size(group=None) -> int:
    if group is None:
        return dist_utils.world_size()
    return dist.get_world_size(group=group)


def split_along_last_dim(tensor: torch.Tensor, group=None) -> torch.Tensor:
    world = _world_size(group)
    width = tensor.size(-1)
    if width % world:
        raise ValueError(f"last dimension {width} must divide across {world} ranks")
    return tensor.chunk(world, dim=-1)[_rank(group)].contiguous()


def gather_along_last_dim(tensor: torch.Tensor, group=None) -> torch.Tensor:
    world = _world_size(group)
    if world == 1:
        return tensor.contiguous()
    pieces = [torch.empty_like(tensor) for _ in range(world)]
    collectives.all_gather(pieces, tensor.contiguous(), group=group)
    return torch.cat(pieces, dim=-1).contiguous()


class CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = grad_output.contiguous().clone()
        if _world_size(ctx.group) > 1:
            collectives.all_reduce(grad, group=ctx.group)
        return grad, None


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        ctx.group = group
        output = tensor.contiguous().clone()
        if _world_size(group) > 1:
            collectives.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class ScatterToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        ctx.group = group
        return split_along_last_dim(tensor, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return gather_along_last_dim(grad_output, ctx.group), None


class GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        ctx.group = group
        return gather_along_last_dim(tensor, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return split_along_last_dim(grad_output, ctx.group), None


def copy_to_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return CopyToTensorParallelRegion.apply(tensor, group)


def reduce_from_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return ReduceFromTensorParallelRegion.apply(tensor, group)


def scatter_to_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return ScatterToTensorParallelRegion.apply(tensor, group)


def gather_from_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return GatherFromTensorParallelRegion.apply(tensor, group)
