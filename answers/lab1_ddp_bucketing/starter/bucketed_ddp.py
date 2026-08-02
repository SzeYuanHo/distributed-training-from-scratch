from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from common import collectives, dist_utils

from .bucket import GradientBucket, assign_buckets

__all__ = ["BucketedDDP"]

DEFAULT_BUCKET_BYTES = 1 << 20


class BucketedDDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        bucket_bytes: int = DEFAULT_BUCKET_BYTES,
        *,
        find_unused_parameters: bool = False,
        overlap: bool = True,
        process_group=None,
    ):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.find_unused_parameters = find_unused_parameters
        self.overlap = overlap
        self._syncing = True
        self._iteration_active = False
        self._next_bucket = 0
        self._unused_ids: set[int] = set()
        self._unused_marked = False
        self._handles = []

        if process_group is None:
            dist_utils.sync_module_params(module)
        elif dist_utils.is_initialized():
            src = dist.get_global_rank(process_group, 0)
            for tensor in list(module.parameters()) + list(module.buffers()):
                collectives.broadcast(tensor.data, src=src, group=process_group)

        seen: set[int] = set()
        self._params: list[nn.Parameter] = []
        for _name, param in module.named_parameters(remove_duplicate=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            self._params.append(param)

        self.buckets = assign_buckets(self._params, bucket_bytes=bucket_bytes, reverse=True)
        self._bucket_of: dict[int, GradientBucket] = {}
        self._world_size = (
            dist.get_world_size(group=process_group) if dist_utils.is_initialized() else 1
        )
        for bucket in self.buckets:
            bucket.process_group = process_group
            bucket.world_size = self._world_size
            for param in bucket.params:
                self._bucket_of[id(param)] = bucket

        self._names: dict[int, str] = {}
        for name, param in module.named_parameters(remove_duplicate=False):
            self._names[id(param)] = name

        for param in self._params:
            # post-accumulate sees final .grad once; register_hook may see partial uses
            self._handles.append(
                param.register_post_accumulate_grad_hook(
                    lambda _parameter, current=param: self._on_grad_ready(current)
                )
            )

    def forward(self, *args, **kwargs):
        self._iteration_active = True
        output = self.module(*args, **kwargs)
        if self.find_unused_parameters and self._syncing:
            self._unused_ids = self._find_unused(output)
            self._unused_marked = False
        return output

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def _find_unused(self, output) -> set[int]:
        roots = []

        def collect(value) -> None:
            if isinstance(value, torch.Tensor):
                if value.grad_fn is not None:
                    roots.append(value.grad_fn)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(output)
        reachable: set[int] = set()
        seen_nodes: set[int] = set()
        stack = roots
        parameter_ids = {id(param) for param in self._params}

        while stack:
            node = stack.pop()
            if node is None or id(node) in seen_nodes:
                continue
            seen_nodes.add(id(node))
            variable = getattr(node, "variable", None)
            if variable is not None and id(variable) in parameter_ids:
                reachable.add(id(variable))
            stack.extend(next_node for next_node, _index in node.next_functions if next_node)

        return parameter_ids - reachable

    def _mark_known_unused(self) -> None:
        if self._unused_marked or not self.find_unused_parameters:
            return
        # Traversing the graph costs a walk per step, which is why real DDP makes
        # it opt-in. The cheaper option, filling unfilled buckets after backward,
        # disagrees between ranks when a parameter is used on some and not others,
        # and then you are back to a hang.
        for param in self._params:
            if id(param) not in self._unused_ids:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            self._bucket_of[id(param)].mark_ready(param)
        self._unused_marked = True

    def _flush_ready_buckets(self) -> None:
        while self._next_bucket < len(self.buckets):
            bucket = self.buckets[self._next_bucket]
            if not bucket.is_full():
                break
            bucket.flush(async_op=self.overlap, world_size=self._world_size)
            self._next_bucket += 1

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        if not self._syncing:
            return
        self._iteration_active = True
        self._mark_known_unused()
        self._bucket_of[id(param)].mark_ready(param)
        self._flush_ready_buckets()

    def finalize(self) -> None:
        if not self.buckets or not self._iteration_active:
            return

        self._mark_known_unused()
        self._flush_ready_buckets()

        incomplete = [bucket.index for bucket in self.buckets if not bucket.is_full()]
        if incomplete:
            raise RuntimeError(
                f"incomplete gradient buckets {incomplete}; "
                "enable find_unused_parameters for conditional graphs"
            )

        for bucket in self.buckets:
            bucket.wait()
        for bucket in self.buckets:
            bucket.reset()

        self._next_bucket = 0
        self._unused_ids.clear()
        self._unused_marked = False
        self._iteration_active = False

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        previous = self._syncing
        self._syncing = False
        try:
            yield
        finally:
            self._syncing = previous

    def bucket_summary(self) -> str:
        lines = []
        for bucket in self.buckets:
            first = self._names.get(id(bucket.params[0]), "?")
            last = self._names.get(id(bucket.params[-1]), "?")
            lines.append(
                f"{bucket.index}: {len(bucket.params)} params, "
                f"{bucket.total_bytes()} bytes, {first} -> {last}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for bucket in self.buckets:
            bucket.reset()
        self._bucket_of.clear()
        self.buckets.clear()
