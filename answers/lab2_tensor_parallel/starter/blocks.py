from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.reference_gpt import GPTConfig

from .layers import ColumnParallelLinear, RowParallelLinear, tp_rank, tp_world_size

__all__ = ["ParallelMLP", "ParallelCausalSelfAttention"]


class ParallelMLP(nn.Module):
    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
            gather_output=False,
            group=group,
        )
        self.gelu = nn.GELU()
        self.c_proj = RowParallelLinear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
            group=group,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    def load_from_reference(self, mlp: nn.Module) -> None:
        self.c_fc.load_from_reference(mlp.c_fc)
        self.c_proj.load_from_reference(mlp.c_proj)


class ParallelCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        world = tp_world_size(group)
        if config.n_head % world:
            raise ValueError(f"n_head={config.n_head} does not divide tp_size={world}")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.local_heads = config.n_head // world
        self.local_hidden = config.n_embd // world
        self.group = group

        self.c_attn = ColumnParallelLinear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
            gather_output=False,
            group=group,
        )
        self.c_proj = RowParallelLinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
            group=group,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.c_attn(x).split(self.local_hidden, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq, self.local_heads, self.head_dim).transpose(1, 2)

        q, k, v = (split_heads(tensor) for tensor in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq, :seq] == 0, float("-inf"))
        probs = self.attn_dropout(F.softmax(scores, dim=-1))
        output = probs @ v
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.local_hidden)
        return self.resid_dropout(self.c_proj(output))

    def load_from_reference(self, attn: nn.Module) -> None:
        rank = tp_rank(self.group)
        start = rank * self.local_heads
        stop = start + self.local_heads

        # c_attn stacks Q, K and V along the output dim, so chunk(t, dim=0) would
        # hand rank 0 the whole of Q. Reshape to (3, heads, head_dim, hidden) and
        # slice the head axis instead, which takes this rank's heads from all three.
        weight = attn.c_attn.weight.view(3, self.n_head, self.head_dim, self.n_embd)
        with torch.no_grad():
            self.c_attn.weight.copy_(weight[:, start:stop].reshape_as(self.c_attn.weight))
            if self.c_attn.bias is not None:
                bias = attn.c_attn.bias.view(3, self.n_head, self.head_dim)
                self.c_attn.bias.copy_(bias[:, start:stop].reshape_as(self.c_attn.bias))
        self.c_proj.load_from_reference(attn.c_proj)
