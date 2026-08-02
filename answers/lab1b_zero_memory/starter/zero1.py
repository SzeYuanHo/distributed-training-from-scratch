from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from common import collectives

__all__ = ["ZeRO1Optimizer", "shard_parameters"]


def shard_parameters(params: list[nn.Parameter], world_size: int) -> list[list[nn.Parameter]]:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    unique = []
    seen = set()
    for param in params:
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            unique.append(param)
    if len(unique) < world_size:
        raise ValueError(
            f"cannot make {world_size} nonempty shards from {len(unique)} trainable parameters"
        )

    # Largest first onto the emptiest rank. Round-robin gives one rank the tied
    # embedding and everyone else the LayerNorms, which is most of the saving gone.
    # Still imperfect while whole tensors cannot be split; real ZeRO shards byte
    # ranges for exactly that reason.
    shards: list[list[nn.Parameter]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for param in sorted(unique, key=lambda item: item.numel() * item.element_size(), reverse=True):
        owner = min(range(world_size), key=lambda rank: loads[rank])
        shards[owner].append(param)
        loads[owner] += param.numel() * param.element_size()
    if any(not shard for shard in shards):
        counts = [len(shard) for shard in shards]
        raise AssertionError(f"parameter shard counts must all be nonzero; got {counts}")
    return shards


class ZeRO1Optimizer:
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        process_group=None,
    ):
        self.params = []
        seen = set()
        for param in params:
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                self.params.append(param)
        if not self.params:
            raise ValueError("ZeRO1Optimizer needs at least one trainable parameter")

        self.process_group = process_group
        self.world_size = (
            torch.distributed.get_world_size(group=process_group)
            if torch.distributed.is_initialized()
            else 1
        )
        self.rank = (
            torch.distributed.get_rank(group=process_group)
            if torch.distributed.is_initialized()
            else 0
        )
        self.shards = shard_parameters(self.params, self.world_size)
        self.local_params = self.shards[self.rank]
        self.inner = torch.optim.AdamW(
            self.local_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.param_groups = self.inner.param_groups
        self.defaults = self.inner.defaults
        self._shard_numel = [sum(param.numel() for param in shard) for shard in self.shards]
        self._width = max(self._shard_numel)
        self._owner = {
            id(param): owner for owner, shard in enumerate(self.shards) for param in shard
        }

    def _empty_segment(self) -> torch.Tensor:
        first = self.params[0]
        return torch.zeros(self._width, dtype=first.dtype, device=first.device)

    def _reduce_scatter_gradients(self) -> None:
        if self.world_size == 1:
            return

        segments = []
        for shard in self.shards:
            segment = self._empty_segment()
            offset = 0
            for param in shard:
                width = param.numel()
                if param.grad is not None:
                    segment[offset : offset + width].copy_(param.grad.reshape(-1))
                offset += width
            segments.append(segment)

        flat = torch.cat(segments)
        reduced = self._empty_segment()
        collectives.reduce_scatter_tensor(reduced, flat, group=self.process_group)
        reduced.div_(self.world_size)

        offset = 0
        for param in self.local_params:
            width = param.numel()
            param.grad = reduced[offset : offset + width].view_as(param)
            offset += width
        for param in self.params:
            if self._owner[id(param)] != self.rank:
                param.grad = None

    def _all_gather_parameters(self) -> None:
        if self.world_size == 1:
            return

        local = self._empty_segment()
        offset = 0
        for param in self.local_params:
            width = param.numel()
            local[offset : offset + width].copy_(param.detach().reshape(-1))
            offset += width

        gathered = torch.empty(
            self.world_size * self._width,
            dtype=local.dtype,
            device=local.device,
        )
        collectives.all_gather_into_tensor(gathered, local, group=self.process_group)
        for owner, shard in enumerate(self.shards):
            segment = gathered[owner * self._width : (owner + 1) * self._width]
            offset = 0
            for param in shard:
                width = param.numel()
                param.data.copy_(segment[offset : offset + width].view_as(param))
                offset += width

    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._reduce_scatter_gradients()
        self.inner.step()
        self._all_gather_parameters()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "optimizer": self.inner.state_dict(),
        }

    def local_state_bytes(self) -> int:
        total = 0
        for state in self.inner.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total
