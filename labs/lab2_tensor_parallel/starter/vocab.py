"""Steps 6 and 7: vocab-parallel embedding and cross-entropy.

The vocabulary is often the largest single dimension in a language model, and the
logit tensor ``(b, s, V)`` is frequently the largest activation anywhere. For
V=128000, b=8, s=512 that is 2 GB in fp32, larger than the parameters of a small
model. Sharding the vocabulary and then gathering the logits to compute the loss
would defeat the entire purpose, so the loss is computed without ever
materialising them.

Three small collectives instead of one enormous all-gather. That is the trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["VocabParallelEmbedding", "vocab_parallel_cross_entropy", "vocab_range"]


def vocab_range(vocab_size: int, rank: int, world_size: int) -> tuple[int, int]:
    """Half-open ``[start, stop)`` of vocabulary indices owned by ``rank``.

    Asserts ``vocab_size % world_size == 0`` with both numbers in the message. The
    reference model pads its vocabulary to a multiple of 8 precisely so this holds;
    ``common/reference_gpt.py`` explains why in its config validator.
    """
    raise NotImplementedError("Step 6a.")


class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary split across ranks.

    Rank ``i`` holds rows ``[start_i, stop_i)`` of the embedding table. A token id
    outside that range is not this rank's problem, and the result is assembled by
    summing across ranks.

    The forward pass, as postconditions rather than code:

    1. Build a mask marking which of the input ids fall in this rank's range.
    2. Shift the in-range ids to local indices, so ``vocab_id - start``.
    3. Look up. Out-of-range ids must not index the table at all: clamp them to
       something valid first, because ``nn.functional.embedding`` with a negative
       or oversized index either raises or, on some backends, reads out of bounds.
    4. Zero the rows corresponding to out-of-range ids. **This is the step people
       skip.** Without it every rank contributes a garbage row and the sum is
       nonsense.
    5. All-reduce. Every rank contributed either the real embedding or zeros, so
       the sum is the real embedding.

    Args:
        num_embeddings: full vocabulary size. Must divide by the group size.
        embedding_dim: model dimension. Not sharded.
        group: process group.

    Attributes the tests use:
        weight: ``(num_embeddings // t, embedding_dim)``
        vocab_start, vocab_end: this rank's range

    Provide ``load_from_reference(embedding)``.

    ⚠ **Weight tying.** The reference model ties ``wte.weight`` and
    ``lm_head.weight``. Under vocab parallelism the embedding is sharded by row and
    the LM head is a column-parallel linear sharded by output dimension, and those
    are the same axis, so the tie can be preserved. Doing so is what makes the
    tied-weight test in ``tp_gpt.py`` pass. Convince yourself the two shardings
    line up before relying on it.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, group=None):
        super().__init__()
        raise NotImplementedError("Step 6b: compute this rank's range, allocate the shard.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Step 6c: mask, shift, look up, zero, all-reduce.")

    def load_from_reference(self, embedding: nn.Embedding) -> None:
        raise NotImplementedError("Step 6d.")


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    targets: torch.Tensor,
    group=None,
) -> torch.Tensor:
    """Cross-entropy over logits sharded along the vocabulary dimension.

    Args:
        local_logits: ``(b, s, V // t)``, this rank's slice of the vocabulary.
        targets: ``(b, s)`` global token ids, identical on every rank.
        group: process group.

    Returns:
        Scalar loss, identical on every rank, matching ``F.cross_entropy`` on the
        full logits to about 1e-5.

    THEORY section 7 derives this. The five steps, as postconditions:

    **1. Global maximum, for numerical stability.** Take the max over this rank's
    vocabulary slice, then all-reduce with MAX. Every rank now has the true global
    maximum for each position. Subtract it. Skipping this is fine on toy logits and
    produces ``inf`` on real ones, where logits reach 30 or more and ``exp(30)``
    overflows fp16 comfortably.

    **2. Global denominator.** Sum ``exp`` over the local slice, all-reduce with
    SUM. That is ``Σ exp(z - max)`` over the full vocabulary.

    **3. The target logit.** The target id lies in exactly one rank's range. That
    rank contributes ``z[target]``; every other rank contributes zero. All-reduce
    with SUM and every rank has the right value. The masking here is the same
    pattern as the embedding, and it is equally easy to forget.

    **4. The loss.** ``log(Σ exp) - z_target``, averaged over positions.

    **5. The backward.** ``softmax(z) - onehot(target)``, and rank ``i`` holds
    exactly the ``V/t`` entries it needs. Work out whether that needs any
    additional communication; THEORY section 7 asks, and the answer is why this
    approach is cheap.

    On implementing the backward: you can write this as a
    ``torch.autograd.Function`` with an explicit backward, which is what Megatron
    does and which is faster. Or you can build it from differentiable ops and let
    autograd handle it, given that every collective you use is already one of your
    mapping ``Function``s. The second is easier to get right. Say which you chose.

    **Collective count:** three in the forward pass, on tensors of shape ``(b, s)``
    rather than ``(b, s, V)``. That ratio is the whole argument. The test asserts
    it.
    """
    raise NotImplementedError("Step 7: the five steps above.")
