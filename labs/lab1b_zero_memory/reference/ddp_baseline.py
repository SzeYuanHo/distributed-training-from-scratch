"""A plain, working DDP for Lab 1B to compare against.

This deliberately does not import Lab 1's ``BucketedDDP``. Labs are meant to be
independently completable, and making 1B depend on a finished Lab 1 would break
that. It is also simpler than Lab 1's version on purpose: no bucketing, no
overlap, nothing to get wrong. The point here is the memory, not the throughput.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import collectives, dist_utils

__all__ = ["PlainDDP", "train_steps", "optimizer_state_bytes"]


class PlainDDP(nn.Module):
    """Broadcast at construction, all-reduce every gradient after backward."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        dist_utils.sync_module_params(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def sync_gradients(self) -> None:
        world = dist_utils.world_size()
        if world == 1:
            return
        seen = set()
        for param in self.module.parameters():
            if param.grad is None or id(param) in seen:
                continue
            seen.add(id(param))
            collectives.all_reduce(param.grad)
            param.grad /= world


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Bytes held in optimizer state. Zero before the first step, by design.

    Adam allocates ``exp_avg`` and ``exp_avg_sq`` lazily on the first ``step()``.
    Measuring before that gives you zero and looks like the sharding worked
    spectacularly well, which is a nice trap to have fallen into once.
    """
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def train_steps(
    model: nn.Module, optimizer, steps: int, *, batch: int = 8, block: int = 128, seed: int = 1337
) -> list[float]:
    """A few training steps against tiny-shakespeare. Returns the loss curve.

    Used by both the DDP baseline and the ZeRO-1 path so the comparison is
    genuinely like for like: same data, same order, same number of steps.
    """
    from common import data as data_mod

    dataset = data_mod.load_dataset()
    rank, world = dist_utils.rank(), dist_utils.world_size()
    losses: list[float] = []

    for step in range(steps):
        x, y = data_mod.get_batch(
            "train",
            batch_size=batch,
            block_size=block,
            rank=rank,
            world_size=world,
            step=step,
            seed=seed,
            dataset=dataset,
        )
        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if hasattr(model, "sync_gradients"):
            model.sync_gradients()

        optimizer.step()
        losses.append(float(loss.detach()))

    return losses
