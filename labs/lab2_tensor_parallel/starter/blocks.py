"""Steps 4 and 5: the MLP and attention blocks.

Both are short once the layers work. The interesting content is the head-sharding
layout in attention, which is more subtle than it looks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.reference_gpt import GPTConfig

__all__ = ["ParallelMLP", "ParallelCausalSelfAttention"]


class ParallelMLP(nn.Module):
    """The transformer feed-forward block, tensor-parallel.

    ``x -> ColumnParallelLinear(h, 4h) -> GELU -> RowParallelLinear(4h, h)``

    **Exactly one all-reduce in the forward pass**, and the test asserts it. That
    is the payoff of the column-then-row ordering: the first layer emits a shard,
    GELU is elementwise so it applies to a shard unchanged, and the second layer
    consumes the shard directly. No communication in between.

    Which means ``gather_output=False`` on the column layer and
    ``input_is_parallel=True`` on the row layer. Set either one the other way and
    you get a correct answer with double the communication, which every numeric
    test will happily accept and the collective-count test will not.

    Provide ``load_from_reference(mlp)`` taking a ``common.reference_gpt.MLP``.
    """

    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        raise NotImplementedError("Step 4a: one column-parallel layer, one row-parallel.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Step 4b.")

    def load_from_reference(self, mlp: nn.Module) -> None:
        raise NotImplementedError("Step 4c.")


class ParallelCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, sharded by head.

    ``x -> ColumnParallelLinear(h, 3h) -> attention over local heads
         -> RowParallelLinear(h, h)``

    One forward all-reduce, same as the MLP.

    ⚠ **The QKV layout is the trap in this lab.**

    ``c_attn`` produces Q, K and V concatenated along the output dimension, so its
    weight is ``(3h, h)`` in torch's layout, laid out as::

        rows   0 ..  h-1   ->  Q for all heads
        rows   h .. 2h-1   ->  K for all heads
        rows  2h .. 3h-1   ->  V for all heads

    Rank ``i`` needs Q, K **and** V for *its own heads*. So a plain
    ``chunk(t, dim=0)`` gives rank 0 all of Q and rank 1 all of K and V, which is
    not remotely what you want. Work out the correct row indices on paper for
    ``h=8, a=2, t=2`` before writing the code; THEORY section 4 asks you to.

    Two ways to handle it:

    * Reshape the reference weight to ``(3, a, head_dim, h)``, slice the head axis,
      and flatten back. Clean and easy to verify.
    * Compute the three index ranges explicitly and concatenate. Faster to write,
      easier to get wrong.

    ⚠ **Whole heads, always.** ``n_head % tp_size == 0``. Assert it with both
    numbers and the preset name in the message, because the ``tiny`` preset has
    four heads and someone will try ``tp_size=8`` on it.

    Why whole heads: attention scores are ``q · k`` over the full ``head_dim``. A
    head split across two ranks would need a collective inside the attention
    computation, per head, per position, which destroys the cost model that makes
    tensor parallelism worth doing at all.

    **The causal mask** is not sharded. Every rank computes attention over the full
    sequence for its own heads, so it needs the whole ``(1, 1, T, T)`` mask.
    Sharding it is a mistake that produces subtly wrong attention at the sequence
    boundaries.

    Provide ``load_from_reference(attn)`` taking a
    ``common.reference_gpt.CausalSelfAttention``.
    """

    def __init__(self, config: GPTConfig, group=None):
        super().__init__()
        raise NotImplementedError(
            "Step 5a: assert head divisibility, build the column-parallel QKV and the "
            "row-parallel output projection, register the full causal mask."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same shapes as the reference, with ``n_head`` replaced by local heads.

        Walk it through:

        * ``x`` is ``(B, T, C)``, replicated across the TP group.
        * ``c_attn(x)`` is ``(B, T, 3C/t)``, this rank's share.
        * split into q, k, v, each ``(B, T, C/t)``.
        * reshape to ``(B, local_heads, T, head_dim)``.
        * attention exactly as the reference does it, no communication.
        * back to ``(B, T, C/t)``.
        * ``c_proj`` reduces and returns ``(B, T, C)``.
        """
        raise NotImplementedError("Step 5b.")

    def load_from_reference(self, attn: nn.Module) -> None:
        """Slice this rank's heads out of an unsharded attention module.

        This is the method the equivalence test exercises hardest. If the full-model
        logits are wrong and the layer tests pass, the bug is almost certainly here.
        """
        raise NotImplementedError("Step 5c: get the QKV row indices right.")
