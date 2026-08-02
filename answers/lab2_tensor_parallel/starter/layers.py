from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import dist_utils

from .mappings import (
    copy_to_tp_region,
    gather_from_tp_region,
    reduce_from_tp_region,
    scatter_to_tp_region,
)

__all__ = ["ColumnParallelLinear", "RowParallelLinear", "tp_rank", "tp_world_size"]


def tp_rank(group=None) -> int:
    if group is None:
        return dist_utils.rank()
    return dist.get_rank(group=group)


def tp_world_size(group=None) -> int:
    if group is None:
        return dist_utils.world_size()
    return dist.get_world_size(group=group)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        group=None,
    ):
        super().__init__()
        world = tp_world_size(group)
        if out_features % world:
            raise ValueError(f"out_features {out_features} must divide across {world} ranks")

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // world
        self.gather_output = gather_output
        self.group = group

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(copy_to_tp_region(x, self.group), self.weight, self.bias)
        if self.gather_output:
            output = gather_from_tp_region(output, self.group)
        return output

    def load_from_reference(self, linear: nn.Linear) -> None:
        expected = (self.out_features, self.in_features)
        got = tuple(linear.weight.shape)
        if got != expected:
            raise ValueError(f"reference weight is {got}, expected {expected}")
        if (self.bias is None) != (linear.bias is None):
            raise ValueError("reference and sharded layer disagree about bias")

        rank = tp_rank(self.group)
        start = rank * self.out_features_per_rank
        stop = start + self.out_features_per_rank
        with torch.no_grad():
            self.weight.copy_(linear.weight[start:stop])
            if self.bias is not None:
                self.bias.copy_(linear.bias[start:stop])


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        group=None,
    ):
        super().__init__()
        world = tp_world_size(group)
        if in_features % world:
            raise ValueError(f"in_features {in_features} must divide across {world} ranks")

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // world
        self.input_is_parallel = input_is_parallel
        self.group = group

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = scatter_to_tp_region(x, self.group)
        output = F.linear(x, self.weight)
        output = reduce_from_tp_region(output, self.group)
        # Bias after the reduce, and unsharded. Adding it before means every rank
        # contributes a copy and the sum carries t of them.
        if self.bias is not None:
            output = output + self.bias
        return output

    def load_from_reference(self, linear: nn.Linear) -> None:
        expected = (self.out_features, self.in_features)
        got = tuple(linear.weight.shape)
        if got != expected:
            raise ValueError(f"reference weight is {got}, expected {expected}")
        if (self.bias is None) != (linear.bias is None):
            raise ValueError("reference and sharded layer disagree about bias")

        rank = tp_rank(self.group)
        start = rank * self.in_features_per_rank
        stop = start + self.in_features_per_rank
        with torch.no_grad():
            self.weight.copy_(linear.weight[:, start:stop])
            if self.bias is not None:
                self.bias.copy_(linear.bias)
