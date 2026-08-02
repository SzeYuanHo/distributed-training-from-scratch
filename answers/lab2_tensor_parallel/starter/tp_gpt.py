from __future__ import annotations

import torch
import torch.nn as nn

from common.reference_gpt import GPT, GPTConfig

from .blocks import ParallelCausalSelfAttention, ParallelMLP
from .layers import ColumnParallelLinear
from .mappings import gather_from_tp_region
from .vocab import VocabParallelEmbedding, vocab_parallel_cross_entropy

__all__ = ["TensorParallelGPT", "ParallelBlock"]


class ParallelBlock(nn.Module):
    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ParallelCausalSelfAttention(config, group)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = ParallelMLP(config, group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

    def load_from_reference(self, block: nn.Module) -> None:
        self.ln_1.load_state_dict(block.ln_1.state_dict())
        self.attn.load_from_reference(block.attn)
        self.ln_2.load_state_dict(block.ln_2.state_dict())
        self.mlp.load_from_reference(block.mlp)


class TensorParallelGPT(nn.Module):
    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        self.config = config
        self.group = group
        self.transformer = nn.ModuleDict(
            {
                "wte": VocabParallelEmbedding(config.vocab_size, config.n_embd, group),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([ParallelBlock(config, group) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = ColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            bias=False,
            gather_output=False,
            group=group,
        )

        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

    @classmethod
    def from_reference(cls, model: GPT, group=None) -> TensorParallelGPT:
        parallel = cls(model.config, group)
        source = next(model.parameters())
        parallel.to(device=source.device, dtype=source.dtype)

        parallel.transformer.wte.load_from_reference(model.transformer.wte)
        parallel.transformer.wpe.load_state_dict(model.transformer.wpe.state_dict())
        for target, reference in zip(parallel.transformer.h, model.transformer.h):
            target.load_from_reference(reference)
        parallel.transformer.ln_f.load_state_dict(model.transformer.ln_f.state_dict())
        parallel.lm_head.load_from_reference(model.lm_head)

        if model.config.tie_weights:
            parallel.lm_head.weight = parallel.transformer.wte.weight
        return parallel

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq = idx.shape
        if seq > self.config.block_size:
            raise ValueError(f"sequence length {seq} exceeds block_size {self.config.block_size}")

        positions = torch.arange(seq, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        local_logits = self.lm_head(x)

        if targets is None:
            return gather_from_tp_region(local_logits, self.group), None

        loss = vocab_parallel_cross_entropy(local_logits, targets, self.group)
        return local_logits, loss

    def collective_count_per_step(self) -> int:
        return 4 * self.config.n_layer + 5
