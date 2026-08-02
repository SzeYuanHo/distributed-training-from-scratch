"""Step 8: assemble the whole thing.

Take a frozen reference GPT, shard its weights, and produce logits that match to
five decimal places. Everything up to now was a component test; this is the one
that says the components compose.

If the layer tests pass and this does not, the bug is almost always in
``load_from_reference`` for attention, where the QKV row indices are easy to get
wrong in a way that still produces valid shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.reference_gpt import GPT, GPTConfig

__all__ = ["TensorParallelGPT", "ParallelBlock"]


class ParallelBlock(nn.Module):
    """One transformer block with parallel attention and MLP.

    The LayerNorms are **not** sharded. They operate on the full hidden dimension,
    which every rank holds because the block's input is replicated. Each rank
    computes the same LayerNorm on the same data, which is redundant work and
    negligible cost.

    That redundancy is exactly what sequence parallelism removes, by sharding the
    LayerNorm regions along the sequence dimension instead. Worth noticing now,
    because it is the stretch goal and it becomes obvious once you have seen the
    waste.
    """

    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        raise NotImplementedError(
            "Step 8a: two LayerNorms, one parallel attention, one parallel MLP."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual, identical structure to ``reference_gpt.Block``."""
        raise NotImplementedError("Step 8b.")

    def load_from_reference(self, block: nn.Module) -> None:
        raise NotImplementedError("Step 8c.")


class TensorParallelGPT(nn.Module):
    """A GPT with tensor parallelism throughout.

    What is sharded and what is not:

    ==========================  ==========================================
    ``wte`` token embedding     vocab-parallel (sharded by row)
    ``wpe`` position embedding  **replicated**, see below
    each block's attention      column-parallel QKV, row-parallel output
    each block's MLP            column-parallel then row-parallel
    ``ln_f``                    replicated
    ``lm_head``                 column-parallel, sharded by vocabulary
    the loss                    vocab-parallel cross-entropy
    ==========================  ==========================================

    **Why the position embedding is not sharded.** It is ``(block_size, n_embd)``,
    which is tiny next to a real vocabulary, and sharding it would need another
    collective per forward for no meaningful saving. Megatron makes the same
    choice. Knowing *why* a thing is not parallelised is as useful as knowing why
    something is.

    **Weight tying.** The reference ties ``wte.weight`` to ``lm_head.weight``.
    Under this layout the embedding is sharded by vocabulary row and the LM head is
    column-parallel sharded by output dimension, and those are the same axis, so
    the tie survives sharding. Preserve it: the test compares against a reference
    that has it, and an untied parallel model will drift after the first optimizer
    step even though its initial logits match.

    Args:
        config: the same ``GPTConfig`` the reference uses.
        group: tensor-parallel process group. None means the world.
    """

    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        raise NotImplementedError("Step 8d.")

    @classmethod
    def from_reference(cls, model: GPT, group=None) -> TensorParallelGPT:
        """Build a sharded model holding this rank's slice of a reference model's weights.

        Every rank is given the *same* full reference model and each takes its own
        slice. That is not how you would load a checkpoint in production, where
        each rank reads only its shard from disk, but it makes the test trivially
        correct: there is one source of truth and no question about whether the
        ranks started from the same weights.

        After this, ``tp_model(x)`` logits must match ``model(x)`` logits to
        ``rtol=1e-5``. That single assertion covers the entire lab.
        """
        raise NotImplementedError("Step 8e: construct, then load_from_reference throughout.")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Same signature as the reference: ``(idx, targets) -> (logits, loss)``.

        Two decisions to make explicit, and the docstring is the only place the
        answer appears:

        **What shape are the returned logits?** Each rank naturally holds
        ``(b, s, V/t)``. The reference returns ``(b, s, V)``. Gathering them costs
        a collective on the largest tensor in the model, which is exactly what
        vocab parallelism exists to avoid. So: gather them when ``targets is None``
        (the caller wants logits and there is no loss path to keep them sharded
        for), and keep them sharded when computing the loss. Document whichever
        rule you pick, because the test needs to know.

        **Which loss?** ``vocab_parallel_cross_entropy`` on the sharded logits. Not
        ``F.cross_entropy`` on gathered ones. The collective-count test checks.
        """
        raise NotImplementedError("Step 8f.")

    def collective_count_per_step(self) -> int:
        """Predicted all-reduces per training step. Not tested; useful for the writeup.

        THEORY section 5 has the per-layer count. Add the embedding and the loss,
        which are easy to forget and which the test does count.
        """
        raise NotImplementedError("Optional.")
