"""Tier 0: the ledger arithmetic.

Pure numbers. Milliseconds. These are the tests that check whether you understood
the ZeRO paper, and they are the reason to write ``ledger_by_hand`` before reading
``common/memory.py``.
"""

from __future__ import annotations

import pytest

from common import memory
from labs.lab1b_zero_memory.starter import ledger

pytestmark = pytest.mark.tier0

PSI = 1_000_000_000  # a round billion, so the arithmetic is easy to check by eye


# --------------------------------------------------------------------------- #
# the famous number
# --------------------------------------------------------------------------- #


def test_mixed_precision_adamw_is_sixteen_bytes_per_param():
    result = ledger.ledger_by_hand(PSI, dtype="bf16", optimizer="adamw", zero_stage=0, world_size=1)
    assert result.total == 16 * PSI, (
        f"expected 16 bytes per parameter, got {result.total / PSI:.1f}.\n"
        f"  params {result.params / PSI:.0f}  grads {result.grads / PSI:.0f}  "
        f"master {result.master_params / PSI:.0f}  optimizer {result.optimizer_states / PSI:.0f}\n"
        "  Should be 2 + 2 + 4 + 8."
    )


def test_fp32_adamw_is_also_sixteen_bytes_per_param():
    """The result that makes people re-read their own table.

    fp32: 4 for parameters, 4 for gradients, no separate master copy because the
    parameters already are one, 8 for Adam. Sixteen again.

    Which means mixed precision does not halve your *state* memory at all. What it
    halves is activations and the bytes on the wire, and it buys tensor-core
    throughput. Worth knowing before you promise someone a 2x memory saving.
    """
    result = ledger.ledger_by_hand(PSI, dtype="fp32", optimizer="adamw", zero_stage=0, world_size=1)
    assert result.total == 16 * PSI, (
        f"expected 16 bytes per parameter for fp32 AdamW, got {result.total / PSI:.1f}. "
        "In fp32 the parameters are the master copy, so there is no separate one."
    )
    assert result.master_params == 0, (
        "fp32 training has no separate master copy; the parameters are already fp32"
    )


@pytest.mark.parametrize(
    "optimizer,expected_per_param",
    [
        ("adamw", 16),
        ("adam", 16),
        ("sgd_momentum", 12),
        ("sgd", 8),
    ],
)
def test_optimizer_choice_changes_the_ledger(optimizer, expected_per_param):
    result = ledger.ledger_by_hand(
        PSI, dtype="bf16", optimizer=optimizer, zero_stage=0, world_size=1
    )
    assert result.total == expected_per_param * PSI, (
        f"{optimizer}: expected {expected_per_param} bytes/param, got {result.total / PSI:.1f}"
    )


# --------------------------------------------------------------------------- #
# the stages
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "stage,expected_gb",
    [
        (0, 16.0),  # 2 + 2 + 12
        (1, 5.5),  # 2 + 2 + 12/8
        (2, 3.75),  # 2 + (2 + 12)/8
        (3, 2.0),  # (2 + 2 + 12)/8
    ],
)
def test_zero_stages_at_eight_ranks(stage, expected_gb):
    """The table from THEORY section 2, as numbers. Ψ = 1e9, N = 8."""
    result = ledger.ledger_by_hand(
        PSI, dtype="bf16", optimizer="adamw", zero_stage=stage, world_size=8
    )
    got_gb = result.total / 1e9
    assert got_gb == pytest.approx(expected_gb, rel=0.01), (
        f"ZeRO-{stage} at N=8 should be {expected_gb} GB per rank, got {got_gb:.2f} GB.\n"
        f"  params {result.params / 1e9:.2f}  grads {result.grads / 1e9:.2f}  "
        f"master {result.master_params / 1e9:.2f}  opt {result.optimizer_states / 1e9:.2f}\n"
        "  Remember the fp32 master copy shards with the optimizer group, not with "
        "the parameters."
    )


def test_stage_three_at_one_rank_equals_stage_zero():
    """Sharding across a world of one is a no-op. A good invariant to have."""
    a = ledger.ledger_by_hand(PSI, zero_stage=0, world_size=1)
    b = ledger.ledger_by_hand(PSI, zero_stage=3, world_size=1)
    assert a.total == b.total


@pytest.mark.parametrize("world", [2, 4, 8, 64])
def test_stages_are_monotonically_cheaper(world):
    totals = [ledger.ledger_by_hand(PSI, zero_stage=s, world_size=world).total for s in range(4)]
    assert totals == sorted(totals, reverse=True), (
        f"at N={world} the stage totals are {[t / 1e9 for t in totals]} GB. "
        "Each stage shards strictly more than the last, so they must not increase."
    )


def test_stage_one_and_two_have_a_floor():
    """The point that decides when stage 3 becomes necessary.

    Stages 1 and 2 never shard the `2Ψ` of parameters, so per-rank memory
    approaches `2Ψ` and stops. Add a thousand ranks and it is still `2Ψ`. Stage 3
    keeps falling.
    """
    huge_n = ledger.ledger_by_hand(PSI, zero_stage=1, world_size=100_000).total
    assert huge_n >= 4 * PSI * 0.99, (
        f"ZeRO-1 at N=100000 came out at {huge_n / PSI:.2f} bytes/param. It should "
        "approach 4 (2 for parameters plus 2 for gradients) and never go below."
    )

    stage3 = ledger.ledger_by_hand(PSI, zero_stage=3, world_size=100_000).total
    assert stage3 < huge_n / 10, "stage 3 shards everything, so it should keep falling"


def test_ledger_matches_the_shared_implementation():
    """Cross-check against common/memory.py, row by row.

    If you wrote yours from the paper, this should pass first time. If it does not,
    the failure names the row, which is more useful than a total mismatch.
    """
    for stage in range(4):
        for world in (1, 4, 8):
            mine = ledger.ledger_by_hand(PSI, "bf16", "adamw", stage, world)
            theirs = memory.analytic_ledger(PSI, "bf16", "adamw", stage, world)

            for field in ("params", "grads", "master_params", "optimizer_states"):
                assert getattr(mine, field) == getattr(theirs, field), (
                    f"stage {stage}, N={world}: {field} is "
                    f"{getattr(mine, field) / PSI:.2f} bytes/param, "
                    f"expected {getattr(theirs, field) / PSI:.2f}"
                )


def test_ledger_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ledger.ledger_by_hand(PSI, dtype="int3")
    with pytest.raises(ValueError):
        ledger.ledger_by_hand(PSI, optimizer="shampoo")
    with pytest.raises(ValueError):
        ledger.ledger_by_hand(PSI, zero_stage=4)


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #


def test_activations_grow_quadratically_with_sequence_length():
    """The `5·a·s/h` term. Double the sequence and this more than doubles.

    It is why long-context training is an activation-memory problem before it is
    anything else, and why the ledger's neat 16Ψ is a floor rather than an answer.
    """
    short = ledger.activation_bytes(8, 256, 384, 6, 8)
    long = ledger.activation_bytes(8, 512, 384, 6, 8)

    ratio = long / short
    assert ratio > 2.2, (
        f"doubling the sequence length grew activations by {ratio:.2f}x. With the "
        "quadratic attention term it should be noticeably more than 2."
    )
    assert ratio < 4.1, f"grew by {ratio:.2f}x, which is more than quadratic; check the formula"


def test_selective_recompute_removes_the_quadratic_term():
    full = ledger.activation_bytes(8, 1024, 384, 6, 8, recompute="none")
    selective = ledger.activation_bytes(8, 1024, 384, 6, 8, recompute="selective")

    assert selective < full, "selective recomputation must reduce activation memory"

    short_full = ledger.activation_bytes(8, 512, 384, 6, 8, recompute="none")
    short_sel = ledger.activation_bytes(8, 512, 384, 6, 8, recompute="selective")

    assert (full / short_full) > (selective / short_sel), (
        "selective recomputation should make activation memory scale more gently "
        "with sequence length, because it drops the quadratic term"
    )


# --------------------------------------------------------------------------- #
# capacity planning
# --------------------------------------------------------------------------- #


def test_fits_in_is_honest_about_headroom():
    eight_gb = 8 * 1024**3

    small_fits, message = ledger.predict_fits(100_000_000, eight_gb)
    assert small_fits, f"100 M parameters should fit in 8 GB: {message}"

    big_fits, message = ledger.predict_fits(2_000_000_000, eight_gb)
    assert not big_fits, "2 B parameters at 16 bytes each is 32 GB and does not fit in 8"
    assert any(token in message.lower() for token in ("short", "need", "gb", "mb")), (
        f"the explanation should say how much is needed and how much is available; got {message!r}"
    )


def test_zero1_raises_the_ceiling():
    eight_gb = 8 * 1024**3
    psi = 700_000_000

    alone, _ = ledger.predict_fits(psi, eight_gb, zero_stage=0, world_size=1)
    sharded, _ = ledger.predict_fits(psi, eight_gb, zero_stage=1, world_size=8)

    assert not alone, "700 M at 16 bytes each is 11.2 GB and should not fit alone"
    assert sharded, "at ZeRO-1 with 8 ranks it is 3.85 GB per rank and should fit"


def test_min_ranks_returns_none_when_the_floor_blocks_you():
    """Stage 1 cannot shard the 4Ψ floor, so past some size no N is enough."""
    eight_gb = 8 * 1024**3

    workable = ledger.min_ranks_for(1_000_000_000, eight_gb, zero_stage=1)
    assert workable is not None and workable > 1

    impossible = ledger.min_ranks_for(50_000_000_000, eight_gb, zero_stage=1)
    assert impossible is None, (
        "50 B parameters needs 200 GB just for parameters and gradients, which "
        "ZeRO-1 never shards. No world size fixes that; return None."
    )

    with_stage3 = ledger.min_ranks_for(50_000_000_000, eight_gb, zero_stage=3)
    assert with_stage3 is not None, "stage 3 shards everything, so some N works"
