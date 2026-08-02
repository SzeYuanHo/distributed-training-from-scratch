from __future__ import annotations

import torch
import torch.nn as nn

from common import collectives
from common.reference_gpt import GPT
from labs.lab2_tensor_parallel.starter.tp_gpt import TensorParallelGPT

__all__ = ["Parallel2DModel", "average_gradients_over_dp"]


def average_gradients_over_dp(model: nn.Module, layout) -> None:
    # DP group only: sharded params are already right within the TP group, and
    # replicated ones got identical gradients there. Reducing over the world
    # would divide by dp*tp and quietly shrink the learning rate.
    # No sequence-parallel regions here, so no replicated param has a gradient
    # that differs across TP.
    if layout.dp_size == 1:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        collectives.all_reduce(param.grad, group=layout.dp_group)
        param.grad.div_(layout.dp_size)


class Parallel2DModel(nn.Module):
    def __init__(self, config, layout):
        super().__init__()
        self.layout = layout
        reference = GPT(config)
        self.model = TensorParallelGPT.from_reference(reference, group=layout.tp_group)

        if layout.dp_size > 1:
            source = layout.tp_rank
            for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                collectives.broadcast(tensor.data, src=source, group=layout.dp_group)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        return self.model(idx, targets)

    def sync_gradients(self) -> None:
        average_gradients_over_dp(self.model, self.layout)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)
