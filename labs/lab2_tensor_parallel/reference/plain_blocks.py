"""Single-process ground truth for Lab 2, one layer at a time.

The reference model in ``common/reference_gpt.py`` is the real target, but testing
against a whole GPT tells you "something is wrong" and not much else. These
factories build one component at a time from the same config, so a failure names
the layer.

Everything here is plain PyTorch. Read it freely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.reference_gpt import GPT, MLP, CausalSelfAttention, GPTConfig

__all__ = [
    "tiny_config",
    "make_linear",
    "make_mlp",
    "make_attention",
    "make_embedding",
    "make_model",
    "reference_cross_entropy",
    "split_weight_columns",
    "split_weight_rows",
]


def tiny_config(**overrides) -> GPTConfig:
    """A config small enough to test quickly and large enough to shard 4 ways."""
    base = dict(
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        vocab_size=32,
        dropout=0.0,
        bias=True,
        max_tp=4,
    )
    base.update(overrides)
    return GPTConfig(**base)


def make_linear(in_features: int, out_features: int, bias: bool = True, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=bias, dtype=torch.float64)


def make_mlp(config: GPTConfig | None = None, seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    return MLP(config or tiny_config()).double()


def make_attention(config: GPTConfig | None = None, seed: int = 0) -> CausalSelfAttention:
    torch.manual_seed(seed)
    return CausalSelfAttention(config or tiny_config()).double()


def make_embedding(num_embeddings: int, dim: int, seed: int = 0) -> nn.Embedding:
    torch.manual_seed(seed)
    return nn.Embedding(num_embeddings, dim, dtype=torch.float64)


def make_model(config: GPTConfig | None = None, seed: int = 0) -> GPT:
    """A full reference GPT. Every rank builds an identical one from the same seed."""
    torch.manual_seed(seed)
    return GPT(config or tiny_config())


def reference_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """What ``vocab_parallel_cross_entropy`` has to match, computed the ordinary way."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# --------------------------------------------------------------------------- #
# hand-sharding helpers for the Tier-0 algebra tests
# --------------------------------------------------------------------------- #


def split_weight_columns(weight: torch.Tensor, n_shards: int) -> list[torch.Tensor]:
    """Split a torch ``(out, in)`` weight along its OUTPUT dimension.

    This is column-parallel sharding. The name says "columns" because the maths
    writes ``W`` as ``(in, out)`` and splits its columns; torch stores the
    transpose, so the split is along dim 0. That mismatch is worth having in one
    place with a comment on it rather than rediscovering in three files.
    """
    return list(weight.chunk(n_shards, dim=0))


def split_weight_rows(weight: torch.Tensor, n_shards: int) -> list[torch.Tensor]:
    """Split a torch ``(out, in)`` weight along its INPUT dimension. Row-parallel."""
    return list(weight.chunk(n_shards, dim=1))
