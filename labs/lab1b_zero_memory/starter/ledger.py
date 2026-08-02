"""Steps 1 and 2: the memory ledger, written from the paper rather than copied.

``common/memory.py`` already contains a working ``analytic_ledger``. The test here
compares your version against it, so copying it makes the test pass immediately.

Please do not. Write it from the ZeRO paper §3 and your own THEORY.md table first,
then run the test and find out whether you had it right. The gap between "I read
the 16Ψ formula" and "I can reconstruct it" is the whole lab, and it is entirely
on the honour system.

Pure arithmetic. No torch, no distribution, tests in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Ledger", "ledger_by_hand", "activation_bytes", "predict_fits", "min_ranks_for", "BYTES"]

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass
class Ledger:
    """Per-rank byte counts. Same field names as ``common.memory.MemoryLedger``.

    Matching the names is deliberate: the test compares field by field and tells
    you which row is wrong, which is more useful than "totals differ".
    """

    params: int = 0
    grads: int = 0
    master_params: int = 0
    optimizer_states: int = 0

    @property
    def total(self) -> int:
        return self.params + self.grads + self.master_params + self.optimizer_states


def ledger_by_hand(
    n_params: int,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    world_size: int = 1,
) -> Ledger:
    """Per-rank memory for one training configuration, from the ZeRO paper.

    Args:
        n_params: Ψ.
        dtype: compute dtype. ``"bf16"`` or ``"fp16"`` means mixed precision, and
            therefore an fp32 master copy. ``"fp32"`` means the parameters *are*
            the master copy, so there is no separate one. That asymmetry is the
            thing THEORY section 1 asks you to notice.
        optimizer: ``"adamw"``, ``"adam"``, ``"sgd"``, or ``"sgd_momentum"``.
            Adam keeps two fp32 state tensors per parameter, SGD with momentum
            keeps one, plain SGD keeps none.
        zero_stage: 0, 1, 2 or 3.
        world_size: N. Sharded components divide by this.

    Returns:
        A :class:`Ledger`.

    The one judgement call: **the fp32 master copy travels with the optimizer
    group**, not with the parameters. It is part of the 12 in "2 + 2 + 12", which
    is why ZeRO-1 shards it even though it is called optimizer-*state* sharding.
    Convince yourself that is right before you code it, because getting it wrong
    gives you `2 + 2 + 8/N + 4` and a ledger that looks plausible.

    Sanity checks the test uses, and you should use them too:

    * bf16 AdamW at stage 0 gives exactly 16 bytes per parameter.
    * fp32 AdamW at stage 0 gives exactly 16 bytes per parameter as well. If your
      answer differs, re-read THEORY section 1.
    * Stage 3 at N=1 equals stage 0 at N=1. Sharding across one rank is a no-op.
    * Totals are monotonically non-increasing in the stage number.

    Raises:
        ValueError: unknown dtype, unknown optimizer, or a stage outside 0-3.
    """
    raise NotImplementedError(
        "Step 1: build the four rows, then shard the right ones for each stage."
    )


def activation_bytes(
    batch_size: int,
    seq_len: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    dtype: str = "bf16",
    recompute: str = "none",
) -> int:
    """Activation memory from Korthikanti et al. §4.

    Per layer, 16-bit, no recomputation, no parallelism::

        s·b·h · (34 + 5·a·s/h)

    ``recompute="full"`` keeps only each layer's input, so `2·s·b·h` per layer, at
    the cost of a second forward pass in the backward.
    ``recompute="selective"`` drops only the `5·a·s/h` term, which is the quadratic
    one, for a fraction of the recompute cost.

    The activation term is missing from the ledger above and it is usually larger.
    That omission is why "my model has 7 B parameters and 16 bytes each, so 112 GB,
    so it fits on two H100s" is a calculation that gets people into trouble.

    Args:
        batch_size: per-rank batch.
        seq_len: sequence length.
        hidden: model dimension.
        n_layers: number of transformer blocks.
        n_heads: attention heads.
        dtype: activation dtype.
        recompute: ``"none"``, ``"selective"``, or ``"full"``.

    Returns:
        Bytes.
    """
    raise NotImplementedError("Step 1b: transcribe the formula from THEORY section 4.")


def predict_fits(
    n_params: int,
    vram_bytes: int,
    *,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    world_size: int = 1,
    activations: int = 0,
    headroom: float = 0.15,
) -> tuple[bool, str]:
    """Will this configuration fit? The actual skill this lab teaches.

    Args:
        n_params: Ψ.
        vram_bytes: device capacity in bytes. Advertised decimal GB and binary GiB
            are different units, and the allocation budget is lower than capacity.
        headroom: fraction reserved for the allocator, workspaces, and
            fragmentation. 15% is a starting guess; a stretch goal is to
            calibrate the real number for your own card by bisecting to OOM.

    Returns:
        ``(fits, explanation)``. The explanation is shown to a human, so make it
        say what is needed, what is available, and by how much it misses. "False"
        on its own is not actionable.
    """
    raise NotImplementedError("Step 2: ledger plus activations against the budget.")


def min_ranks_for(
    n_params: int,
    vram_bytes: int,
    *,
    zero_stage: int = 1,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    activations_per_rank: int = 0,
    headroom: float = 0.15,
    max_ranks: int = 1024,
) -> int | None:
    """Smallest N at which this model fits, or None if no N in range works.

    Note what "no N works" means, and it is worth internalising: at stages 1 and 2
    the unsharded `2Ψ + 2Ψ` never shrinks no matter how many ranks you add, so
    there is a model size beyond which stage 1 cannot help at all and you have to
    move to stage 3 or to tensor parallelism.

    Returns:
        The smallest workable N, or None.
    """
    raise NotImplementedError("Step 2b: search upward, and return None when the floor blocks you.")
