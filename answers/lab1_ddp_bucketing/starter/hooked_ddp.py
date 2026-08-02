from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn

from common import collectives, dist_utils

__all__ = ["HookedDDP"]


class HookedDDP(nn.Module):
    def __init__(self, module: nn.Module, process_group=None):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self._handles = []

        if process_group is None:
            dist_utils.sync_module_params(module)
        elif dist_utils.is_initialized():
            src = dist.get_global_rank(process_group, 0)
            for tensor in list(module.parameters()) + list(module.buffers()):
                collectives.broadcast(tensor.data, src=src, group=process_group)

        seen: set[int] = set()
        for _name, param in module.named_parameters(remove_duplicate=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            # post-accumulate sees final .grad once; register_hook may see partial uses
            self._handles.append(param.register_post_accumulate_grad_hook(self._make_hook(param)))

    def _make_hook(self, param: nn.Parameter):
        def sync(_parameter: nn.Parameter) -> None:
            if param.grad is None or not dist_utils.is_initialized():
                return
            world = dist.get_world_size(group=self.process_group)
            collectives.all_reduce(param.grad, group=self.process_group)
            param.grad.div_(world)

        return sync

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
