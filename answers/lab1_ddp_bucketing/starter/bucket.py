from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from common import collectives, dist_utils

__all__ = ["GradientBucket", "assign_buckets", "param_bytes"]


def param_bytes(param: nn.Parameter) -> int:
    return param.numel() * param.element_size()


class GradientBucket:
    def __init__(self, params: list[nn.Parameter], index: int = 0):
        self.params = list(params)
        self.index = index
        self.ready: set[int] = set()
        self.handle = None
        self.process_group = None
        self.world_size: int | None = None
        self._flat: torch.Tensor | None = None
        self._grads: list[torch.Tensor] = []
        self._flushed = False

    def total_bytes(self) -> int:
        return sum(param_bytes(param) for param in self.params)

    def mark_ready(self, param: nn.Parameter) -> None:
        if any(param is member for member in self.params):
            self.ready.add(id(param))

    def is_full(self) -> bool:
        return len(self.ready) == len(self.params)

    def _finish(self) -> None:
        if self._flat is None:
            return
        self._flat.div_(self.world_size or 1)
        synced = torch._utils._unflatten_dense_tensors(self._flat, self._grads)
        for grad, value in zip(self._grads, synced):
            grad.copy_(value)
        self._flat = None
        self._grads = []

    def flush(self, async_op: bool = False, world_size: int | None = None):
        if self._flushed:
            return self.handle
        if not self.is_full():
            raise RuntimeError(f"bucket {self.index} is not ready")

        grads = [param.grad for param in self.params]
        if any(grad is None for grad in grads):
            raise RuntimeError(f"bucket {self.index} contains a missing gradient")

        self._grads = [grad for grad in grads if grad is not None]
        self._flat = torch._utils._flatten_dense_tensors(self._grads)
        if world_size is not None:
            self.world_size = world_size
        elif self.world_size is None:
            if dist_utils.is_initialized():
                self.world_size = dist.get_world_size(group=self.process_group)
            else:
                self.world_size = 1

        self._flushed = True
        if dist_utils.is_initialized() and (self.world_size or 1) > 1:
            self.handle = collectives.all_reduce(
                self._flat,
                group=self.process_group,
                async_op=async_op,
            )
        else:
            self.handle = None

        if async_op and self.handle is not None:
            return self.handle

        self._finish()
        return None

    def wait(self) -> None:
        if self.handle is not None:
            self.handle.wait()
            self.handle = None
            self._finish()

    def reset(self) -> None:
        self.ready.clear()
        self.handle = None
        self._flat = None
        self._grads = []
        self._flushed = False


def assign_buckets(
    params: list[nn.Parameter], bucket_bytes: int = 1 << 20, reverse: bool = True
) -> list[GradientBucket]:
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")

    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        identity = id(param)
        if not param.requires_grad or identity in seen:
            continue
        seen.add(identity)
        unique.append(param)

    ordered = list(reversed(unique)) if reverse else unique
    groups: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    current_bytes = 0

    for param in ordered:
        size = param_bytes(param)
        if size > bucket_bytes:
            if current:
                groups.append(current)
                current = []
                current_bytes = 0
            groups.append([param])
            continue

        if current and current_bytes + size > bucket_bytes:
            groups.append(current)
            current = []
            current_bytes = 0

        current.append(param)
        current_bytes += size
        if current_bytes == bucket_bytes:
            groups.append(current)
            current = []
            current_bytes = 0

    if current:
        groups.append(current)

    return [GradientBucket(group, index=index) for index, group in enumerate(groups)]
