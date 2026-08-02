"""Ground truth for Lab 3: the whole model in one process.

The pipeline's job is to compute exactly this, spread across ranks. Any
difference is a bug in the pipeline, never a property of pipelining, because
pipelining does not change the arithmetic at all. It only changes who does which
part and when.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common import data as data_mod
from common import reference_gpt

__all__ = ["PipeConfig", "build_model", "run_reference", "run_microbatched_reference"]


@dataclass
class PipeConfig:
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
    block_size: int = 32
    vocab_size: int = 32
    batch_size: int = 8
    n_micro: int = 4
    seed: int = 1337
    lr: float = 1e-3

    @property
    def micro_batch_size(self) -> int:
        if self.batch_size % self.n_micro:
            raise ValueError(f"batch_size {self.batch_size} must divide by n_micro {self.n_micro}")
        return self.batch_size // self.n_micro

    def to_gpt_config(self) -> reference_gpt.GPTConfig:
        return reference_gpt.GPTConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            dropout=0.0,
            bias=True,
            # Untied, because a depth split puts wte and lm_head on different
            # stages and a tied weight would need cross-stage gradient
            # synchronisation. The lab notes this rather than solving it.
            tie_weights=False,
            max_tp=4,
        )


def build_model(config: PipeConfig) -> reference_gpt.GPT:
    torch.manual_seed(config.seed)
    return reference_gpt.GPT(config.to_gpt_config())


def get_batch(config: PipeConfig, step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """One global batch. Every rank calls this and slices its own micro-batches out."""
    dataset = data_mod.load_dataset()
    x, y = data_mod.get_batch(
        "train",
        batch_size=config.batch_size,
        block_size=config.block_size,
        rank=0,
        world_size=1,
        step=step,
        seed=config.seed,
        dataset=dataset,
    )
    return x % config.vocab_size, y % config.vocab_size


def run_reference(config: PipeConfig, steps: int = 1) -> dict:
    """Train the whole model in one process. Returns losses and step-0 gradients."""
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    losses = []
    grads = {}

    for step in range(steps):
        x, y = get_batch(config, step)
        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        optimizer.step()
        losses.append(float(loss.detach()))

    return {"losses": losses, "grads": grads, "n_params": model.num_params()}


def run_microbatched_reference(config: PipeConfig, steps: int = 1) -> dict:
    """The same training, split into micro-batches, still in one process.

    This is the intermediate the pipeline actually has to match. Micro-batching
    changes the floating-point summation order relative to one large batch, so the
    losses differ in the last few digits. Comparing the pipeline against *this*
    rather than against ``run_reference`` separates "my pipeline is wrong" from
    "micro-batching reorders additions", which are two different conversations.
    """
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    losses = []
    grads = {}
    micro = config.micro_batch_size

    for step in range(steps):
        x, y = get_batch(config, step)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0

        for i in range(config.n_micro):
            chunk = slice(i * micro, (i + 1) * micro)
            _logits, loss = model(x[chunk], y[chunk])
            (loss / config.n_micro).backward()
            total += float(loss.detach()) / config.n_micro

        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }

        optimizer.step()
        losses.append(total)

    return {"losses": losses, "grads": grads, "n_params": model.num_params()}
