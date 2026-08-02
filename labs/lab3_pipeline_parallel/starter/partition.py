"""Step 1: split the model by depth.

Pipeline parallelism divides a model along its *layers* rather than along its
tensors. Stage 0 holds the embedding and the first few blocks, the last stage
holds the final blocks plus the LM head and the loss, and each stage owns only its
own parameters.

Everything here is Tier 0. No communication, one process, and the test runs the
stages in sequence and compares against the full model. Get this solid before any
message is sent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.reference_gpt import GPT

__all__ = ["PipelineStage", "split_model", "stage_flops", "balanced_split_points"]


class PipelineStage(nn.Module):
    """One contiguous depth-slice of the model.

    Owns only its own layers, so ``stage.parameters()`` is a strict subset of the
    full model's and the four stages together partition it exactly. That is what
    makes pipeline parallelism a memory win as well as a compute one.

    Args:
        layers: the modules this stage runs, in order.
        stage_index: position in the pipeline. 0 is first.
        n_stages: total.
        is_first: this stage owns the embeddings and consumes token ids.
        is_last: this stage owns the final LayerNorm, the LM head, and the loss.

    The first and last stages are genuinely different from the middle ones, and
    pretending otherwise is where most of the fiddliness in real implementations
    comes from:

    * The first stage's input is ``(B, T)`` integer token ids. Every other stage's
      input is ``(B, T, C)`` floats. Different shape, different dtype.
    * The last stage's output is a loss scalar when targets are supplied, and
      logits otherwise. Every other stage's output is ``(B, T, C)`` floats.

    So a receiver needs to know which case it is in before it can allocate a
    buffer, and it cannot ask. That is why ``activation_shape`` exists.
    """

    def __init__(
        self,
        layers: list[nn.Module],
        stage_index: int,
        n_stages: int,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("Step 1a: store the layers as a ModuleList plus the flags.")

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """Run this stage's layers.

        First stage: ``x`` is token ids, so embed them first.
        Last stage: apply the final norm and head, and compute the loss when
        ``targets`` is given.
        Middle stages: run the blocks and pass the hidden state on.
        """
        raise NotImplementedError("Step 1b.")

    def activation_shape(self, batch_size: int, seq_len: int, hidden: int) -> tuple:
        """Shape of the tensor this stage *sends* to the next one.

        The receiver calls this to size its buffer, because there is no shape
        negotiation in ``recv``. Getting it wrong produces a hang or silently
        reinterpreted bytes, neither of which raises.
        """
        raise NotImplementedError("Step 1c.")


def split_model(
    model: GPT, n_stages: int, split_points: list[int] | None = None
) -> list[PipelineStage]:
    """Cut a reference GPT into ``n_stages`` contiguous stages.

    Args:
        model: a ``common.reference_gpt.GPT``.
        n_stages: how many pieces.
        split_points: layer indices where the cuts fall. When None, divide evenly
            by layer count, which is the naive choice THEORY section 5 asks you to
            criticise.

    Returns:
        Stages in pipeline order.

    What goes where:

    * Stage 0 gets ``wte``, ``wpe``, ``drop``, and blocks ``0 .. k_0``.
    * Middle stages get their blocks and nothing else.
    * The last stage gets its blocks, ``ln_f``, and ``lm_head``.

    ⚠ **Weight tying breaks under a depth split.** The reference ties ``wte.weight``
    to ``lm_head.weight``, and those now live on different stages. Real pipeline
    implementations either replicate the tied weight on both stages and all-reduce
    its gradient, or drop the tie. For this lab, note which you did and why; the
    equivalence test uses an untied config so it does not force your hand, but the
    problem is real and knowing it exists matters more than the fix.

    Assert that the stages together contain every parameter of the original model
    exactly once. A parameter that ends up on two stages gets two gradients and no
    coordination between them.

    Raises:
        ValueError: if ``n_stages`` exceeds the layer count, or if ``split_points``
            is not sorted and within range.
    """
    raise NotImplementedError("Step 1d.")


def stage_flops(stage: PipelineStage, batch: int, seq: int, hidden: int, vocab: int) -> int:
    """Estimated forward FLOPs for one micro-batch through this stage.

    Not for correctness. For seeing the imbalance THEORY section 5 predicts,
    which you cannot fix until you can measure it.

    Rough per-token costs:

    * transformer block: ``24·h²`` plus the attention quadratic term ``4·s·h``
    * LM head: ``2·h·V``
    * embedding lookup: negligible, it is a gather

    With a small vocabulary the last stage looks cheap. With ``V=128000`` the LM
    head alone can exceed a transformer block, and the last stage becomes the
    bottleneck for the whole pipeline.
    """
    raise NotImplementedError("Step 1e.")


def balanced_split_points(model: GPT, n_stages: int, batch: int, seq: int) -> list[int]:
    """Split points that balance estimated FLOPs rather than layer count.

    Stretch goal. Naive splitting gives every stage the same number of blocks;
    this gives them the same amount of work, which means fewer blocks on the first
    and last stages.

    Returns:
        Layer indices, suitable for ``split_model(..., split_points=...)``.
    """
    raise NotImplementedError("Stretch goal, not required by the tests.")
