from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.reference_gpt import GPT

__all__ = ["PipelineStage", "split_model", "stage_flops", "balanced_split_points"]


class PipelineStage(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        stage_index: int,
        n_stages: int,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.stage_index = stage_index
        self.n_stages = n_stages
        self.is_first = is_first
        self.is_last = is_last
        self.wte = None
        self.wpe = None
        self.drop = None
        self.ln_f = None
        self.lm_head = None
        self.hidden_size = 0

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        if self.is_first:
            pos = torch.arange(x.size(1), device=x.device)
            x = self.drop(self.wte(x) + self.wpe(pos))

        for layer in self.layers:
            x = layer(x)

        if not self.is_last:
            return x

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def activation_shape(self, batch_size: int, seq_len: int, hidden: int) -> tuple:
        return (batch_size, seq_len, hidden)


def _boundaries(n_layers: int, n_stages: int, split_points: list[int] | None) -> list[int]:
    if n_stages < 1 or n_stages > n_layers:
        raise ValueError(f"cannot split {n_layers} layers across {n_stages} stages")

    if split_points is None:
        base, extra = divmod(n_layers, n_stages)
        sizes = [base + (i < extra) for i in range(n_stages)]
        cuts = [0]
        for size in sizes:
            cuts.append(cuts[-1] + size)
        return cuts

    if len(split_points) != n_stages - 1:
        raise ValueError(f"expected {n_stages - 1} split points, got {len(split_points)}")
    if split_points != sorted(set(split_points)):
        raise ValueError("split points must be strictly increasing")
    if any(point <= 0 or point >= n_layers for point in split_points):
        raise ValueError(f"split points must lie between 1 and {n_layers - 1}")
    return [0, *split_points, n_layers]


def split_model(
    model: GPT, n_stages: int, split_points: list[int] | None = None
) -> list[PipelineStage]:
    layers = list(model.transformer.h)
    cuts = _boundaries(len(layers), n_stages, split_points)

    if n_stages > 1 and model.lm_head.weight is model.transformer.wte.weight:
        # the two ends cannot keep a local weight tie
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())

    stages = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        stage = PipelineStage(
            layers[start:end],
            index,
            n_stages,
            is_first=index == 0,
            is_last=index == n_stages - 1,
        )
        stage.hidden_size = model.config.n_embd
        stages.append(stage)

    first, last = stages[0], stages[-1]
    first.wte = model.transformer.wte
    first.wpe = model.transformer.wpe
    first.drop = model.transformer.drop
    last.ln_f = model.transformer.ln_f
    last.lm_head = model.lm_head

    original = {id(param) for param in model.parameters()}
    assigned = [id(param) for stage in stages for param in stage.parameters()]
    if len(assigned) != len(set(assigned)) or set(assigned) != original:
        raise RuntimeError("pipeline stages do not partition the model parameters")
    return stages


def stage_flops(stage: PipelineStage, batch: int, seq: int, hidden: int, vocab: int) -> int:
    per_token = len(stage.layers) * (24 * hidden**2 + 4 * seq * hidden)
    if stage.is_last:
        per_token += 2 * hidden * vocab
    return int(batch * seq * per_token)


def balanced_split_points(model: GPT, n_stages: int, batch: int, seq: int) -> list[int]:
    n_layers = len(model.transformer.h)
    if n_stages < 1 or n_stages > n_layers:
        raise ValueError(f"cannot split {n_layers} layers across {n_stages} stages")
    if n_stages == 1:
        return []

    block = batch * seq * (24 * model.config.n_embd**2 + 4 * seq * model.config.n_embd)
    head = batch * seq * 2 * model.config.n_embd * model.config.vocab_size
    best = None
    for points in combinations(range(1, n_layers), n_stages - 1):
        cuts = (0, *points, n_layers)
        costs = [(cuts[i + 1] - cuts[i]) * block for i in range(n_stages)]
        costs[-1] += head
        score = (max(costs), max(costs) - min(costs))
        if best is None or score < best[0]:
            best = (score, list(points))
    return best[1]
