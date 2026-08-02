"""Ground truth: one process, one big batch, no communication anywhere.

Every Tier-1 test in this lab compares against this. The comparison is the whole
design: a loss curve that falls is not evidence of anything, and this file is what
turns "it seems to work" into "it produces the same numbers".

The equivalence being tested is precise, so it is worth stating exactly:

    N ranks, each with per-rank batch b, is equivalent to
    1 process with batch N*b,

given identical initial weights, the same data, and gradients *averaged* rather
than summed. The batch construction in ``common/data.py`` guarantees the data half
by building the global batch first and slicing it, so any mismatch is yours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common import data as data_mod
from common import reference_gpt

__all__ = ["TrainConfig", "ReferenceRun", "train_reference", "build_model"]


@dataclass
class TrainConfig:
    """Shared knobs, so the reference and your DDP cannot silently disagree."""

    preset: str = "tiny"
    steps: int = 20
    per_rank_batch: int = 4
    world_size: int = 1
    block_size: int = 64
    lr: float = 1e-3
    seed: int = 1337
    weight_decay: float = 0.0  # off by default: it is one fewer thing to match
    betas: tuple[float, float] = (0.9, 0.95)

    @property
    def global_batch(self) -> int:
        return self.per_rank_batch * self.world_size


@dataclass
class ReferenceRun:
    losses: list[float] = field(default_factory=list)
    grad_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    final_params: dict[str, torch.Tensor] = field(default_factory=dict)
    n_params: int = 0


def build_model(config: TrainConfig) -> reference_gpt.GPT:
    """Build the model deterministically.

    Every rank calls this with the same seed and gets bit-identical weights, which
    is why the tests can compare against a single-process run at all. Note that
    relying on that in *your* DDP wrapper would be the desync bug: broadcast
    anyway, because a real model has RNG consumption you do not control.
    """
    torch.manual_seed(config.seed)
    dataset = data_mod.load_dataset()
    model = reference_gpt.make_model(
        config.preset,
        block_size=config.block_size,
        vocab_size=dataset.vocab_size,
    )
    return model


def train_reference(config: TrainConfig | None = None) -> ReferenceRun:
    """Train single-process on the global batch. Returns losses and a gradient snapshot.

    The gradient snapshot is taken after the *first* backward pass, before any
    optimizer step. That makes it a clean test of gradient averaging on its own,
    with no optimizer state to confuse things. If your DDP matches this snapshot
    but the loss curves diverge later, the bug is in the optimizer path, not the
    communication.
    """
    config = config or TrainConfig()

    model = build_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    run = ReferenceRun(n_params=model.num_params())
    dataset = data_mod.load_dataset()

    for step in range(config.steps):
        x, y = data_mod.get_batch(
            "train",
            batch_size=config.global_batch,
            block_size=config.block_size,
            rank=0,
            world_size=1,
            step=step,
            seed=config.seed,
            dataset=dataset,
        )

        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step == 0:
            run.grad_snapshot = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

        optimizer.step()
        run.losses.append(float(loss.detach()))

    run.final_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    return run


def reference_gradients_only(config: TrainConfig | None = None) -> dict[str, torch.Tensor]:
    """Just the step-0 gradients. Cheaper than a full run when that is all you need."""
    config = config or TrainConfig(steps=1)
    return train_reference(config).grad_snapshot


class UnusedBranchModel(torch.nn.Module):
    """A model with a branch that is not always used. Step 8's test subject.

    ``head_b`` receives a gradient only when ``use_b`` is True. Under DDP with
    ``use_b=False``, ``head_b``'s hook never fires, its bucket never fills, and
    ``finalize()`` waits for a collective nobody will issue.

    Small and obvious on purpose. In production this shows up as a
    detection-head that only runs when there are objects in the frame, or a
    mixture-of-experts router that skipped an expert this batch, and it takes an
    afternoon to find.
    """

    def __init__(self, dim: int = 32, out: int = 8):
        super().__init__()
        self.trunk = torch.nn.Linear(dim, dim)
        self.head_a = torch.nn.Linear(dim, out)
        self.head_b = torch.nn.Linear(dim, out)  # gets no gradient when use_b=False

    def forward(self, x: torch.Tensor, use_b: bool = False) -> torch.Tensor:
        hidden = torch.relu(self.trunk(x))
        out = self.head_a(hidden)
        if use_b:
            out = out + self.head_b(hidden)
        return out
