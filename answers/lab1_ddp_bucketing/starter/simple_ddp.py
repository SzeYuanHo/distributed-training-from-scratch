from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn

from common import collectives, dist_utils

__all__ = ["SimpleDDP"]


class SimpleDDP(nn.Module):
    def __init__(self, module: nn.Module, process_group=None):
        super().__init__()
        self.module = module
        self.process_group = process_group

        if process_group is None:
            dist_utils.sync_module_params(module)
        elif dist_utils.is_initialized():
            src = dist.get_global_rank(process_group, 0)
            for tensor in list(module.parameters()) + list(module.buffers()):
                collectives.broadcast(tensor.data, src=src, group=process_group)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_gradients(self) -> None:
        if not dist_utils.is_initialized():
            return
        world = dist.get_world_size(group=self.process_group)
        for param in self.module.parameters():
            if param.grad is None:
                continue
            collectives.all_reduce(param.grad, group=self.process_group)
            param.grad.div_(world)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
