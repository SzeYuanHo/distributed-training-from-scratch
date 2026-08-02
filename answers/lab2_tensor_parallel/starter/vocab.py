from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import collectives

from .layers import tp_rank, tp_world_size
from .mappings import reduce_from_tp_region

__all__ = ["VocabParallelEmbedding", "vocab_parallel_cross_entropy", "vocab_range"]


def vocab_range(vocab_size: int, rank: int, world_size: int) -> tuple[int, int]:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if vocab_size % world_size:
        raise ValueError(f"vocab_size {vocab_size} must divide across {world_size} ranks")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    per_rank = vocab_size // world_size
    return rank * per_rank, (rank + 1) * per_rank


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, group=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group = group

        world = tp_world_size(group)
        self.vocab_start, self.vocab_end = vocab_range(num_embeddings, tp_rank(group), world)
        self.num_embeddings_per_rank = self.vocab_end - self.vocab_start
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_rank, embedding_dim))
        nn.init.normal_(self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        in_range = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(0, self.num_embeddings_per_rank - 1)
        output = F.embedding(local_ids, self.weight)
        output = output * in_range.unsqueeze(-1)
        return reduce_from_tp_region(output, self.group)

    def load_from_reference(self, embedding: nn.Embedding) -> None:
        expected = (self.num_embeddings, self.embedding_dim)
        got = tuple(embedding.weight.shape)
        if got != expected:
            raise ValueError(f"reference embedding is {got}, expected {expected}")
        with torch.no_grad():
            self.weight.copy_(embedding.weight[self.vocab_start : self.vocab_end])


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    targets: torch.Tensor,
    group=None,
) -> torch.Tensor:
    world = tp_world_size(group)
    rank = tp_rank(group)
    local_vocab = local_logits.size(-1)
    vocab_size = local_vocab * world
    start, stop = vocab_range(vocab_size, rank, world)

    global_max = local_logits.max(dim=-1).values.detach().clone()
    if world > 1:
        collectives.all_reduce(global_max, op=collectives.ReduceOp.MAX, group=group)
    shifted = local_logits - global_max.unsqueeze(-1)

    denominator = reduce_from_tp_region(shifted.exp().sum(dim=-1), group)

    valid = targets != -1
    owned = valid & (targets >= start) & (targets < stop)
    local_targets = (targets - start).clamp(0, local_vocab - 1)
    target_logits = local_logits.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1)
    target_logits = reduce_from_tp_region(target_logits * owned, group)

    # Built from differentiable ops instead of a custom autograd.Function. The
    # collectives above already carry their own backward, so autograd arrives at
    # softmax - onehot on its own and there is nothing left to hand-derive.
    # Megatron writes the Function; this was easier to get right.
    losses = denominator.log() + global_max - target_logits
    return losses[valid].mean()
