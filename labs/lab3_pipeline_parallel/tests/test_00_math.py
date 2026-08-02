"""Tier 0: bubble arithmetic and schedule ordering, with no tensors involved.

Two things get tested here, and both catch bugs that would otherwise present as a
hang in a spawned run:

* the partition, run sequentially in one process
* the schedules, as pure sequences of operations

An off-by-one in the 1F1B warmup count deadlocks a real pipeline with no error
message. Here it fails in a millisecond and prints both sequences.
"""

from __future__ import annotations

import pytest
import torch

from labs.lab3_pipeline_parallel.reference import single_process as ref
from labs.lab3_pipeline_parallel.starter import partition, schedule_1f1b, schedule_afab

pytestmark = pytest.mark.tier0


# --------------------------------------------------------------------------- #
# partitioning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_stages", [1, 2, 4])
def test_stage_partition_equivalence(n_stages):
    """Running the stages in sequence must equal the full model's forward."""
    config = ref.PipeConfig(n_layer=4)
    model = ref.build_model(config)
    x, y = ref.get_batch(config)

    expected_logits, expected_loss = model(x, y)

    stages = partition.split_model(model, n_stages)
    assert len(stages) == n_stages

    hidden = x
    loss = None
    for i, stage in enumerate(stages):
        if i == n_stages - 1:
            hidden, loss = stage(hidden, targets=y)
        else:
            hidden = stage(hidden)

    torch.testing.assert_close(
        loss,
        expected_loss,
        rtol=1e-6,
        atol=1e-8,
        msg=f"the {n_stages}-stage split does not reproduce the full model's loss",
    )


@pytest.mark.parametrize("n_stages", [2, 4])
def test_partition_covers_every_parameter_once(n_stages):
    """A parameter on two stages gets two uncoordinated gradients."""
    config = ref.PipeConfig(n_layer=4)
    model = ref.build_model(config)
    stages = partition.split_model(model, n_stages)

    original = {id(p) for p in model.parameters()}
    seen = [id(p) for stage in stages for p in stage.parameters()]

    assert len(seen) == len(set(seen)), (
        "a parameter appears on more than one stage. Each would compute its own "
        "gradient with no synchronisation between them."
    )
    assert set(seen) == original, (
        f"the stages hold {len(set(seen))} parameters, the model has {len(original)}. "
        "Every parameter must land on exactly one stage."
    )


def test_partition_rejects_more_stages_than_layers():
    config = ref.PipeConfig(n_layer=4)
    model = ref.build_model(config)
    with pytest.raises(ValueError):
        partition.split_model(model, 16)


def test_first_and_last_stages_are_flagged():
    config = ref.PipeConfig(n_layer=4)
    stages = partition.split_model(ref.build_model(config), 4)

    assert stages[0].is_first and not stages[0].is_last
    assert stages[-1].is_last and not stages[-1].is_first
    assert not stages[1].is_first and not stages[1].is_last


# --------------------------------------------------------------------------- #
# bubble arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "p,m,expected_overhead,expected_fraction",
    [
        (2, 1, 1.0, 0.5),
        (2, 4, 0.25, 0.2),
        (4, 4, 0.75, 3 / 7),
        (4, 8, 0.375, 3 / 11),
        (4, 16, 0.1875, 3 / 19),
        (8, 16, 0.4375, 7 / 23),
    ],
)
def test_bubble_formulas_use_their_stated_denominators(p, m, expected_overhead, expected_fraction):
    """Check both common bubble quantities from THEORY section 1.

    ``(p-1)/m`` is overhead relative to ideal work. The elapsed-time fraction
    plotted by EventLog is ``(p-1)/(m+p-1)``.
    """
    assert (p - 1) / m == pytest.approx(expected_overhead)
    assert (p - 1) / (m + p - 1) == pytest.approx(expected_fraction)


def test_bubble_falls_as_microbatches_rise():
    fractions = [(4 - 1) / (m + 4 - 1) for m in (1, 2, 4, 8, 16, 32)]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[-1] < 0.1, "at m=32 with p=4 the bubble should be under 10%"


# --------------------------------------------------------------------------- #
# schedules as sequences
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_micro", [1, 4, 8])
def test_afab_schedule_shape(n_micro):
    ops = schedule_afab.afab_schedule(stage_index=0, n_stages=4, n_micro=n_micro)

    assert len(ops) == 2 * n_micro
    kinds = [op.kind for op in ops]
    assert kinds == ["F"] * n_micro + ["B"] * n_micro, (
        f"AFAB runs every forward then every backward; got {''.join(kinds)}"
    )
    assert sorted(op.micro for op in ops if op.kind == "F") == list(range(n_micro))
    assert sorted(op.micro for op in ops if op.kind == "B") == list(range(n_micro))


@pytest.mark.parametrize("stage,expected", [(0, 4), (1, 3), (2, 2), (3, 1)])
def test_1f1b_warmup_counts(stage, expected):
    """The number that decides whether the pipeline runs or deadlocks.

    The last stage warms up for one forward: it can start its backward as soon as
    its own forward completes. Stage 0 warms up for `p`, because the first
    micro-batch has to reach the end and its gradient has to come all the way
    back.
    """
    got = schedule_1f1b.warmup_steps(stage_index=stage, n_stages=4, n_micro=8)
    assert got == expected, (
        f"stage {stage} of 4 should warm up for {expected} forwards, got {got}. "
        "An off-by-one here deadlocks a real pipeline with no error message."
    )


def test_1f1b_warmup_is_capped_by_microbatch_count():
    """With fewer micro-batches than stages there is no steady state at all."""
    got = schedule_1f1b.warmup_steps(stage_index=0, n_stages=8, n_micro=2)
    assert got <= 2, (
        f"warmup of {got} exceeds the 2 available micro-batches. The schedule "
        "degenerates to AFAB here; handle it rather than scheduling work that "
        "does not exist."
    )


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_1f1b_schedule_does_the_same_work_as_afab(stage):
    """Same operations, different order. That is the entire difference."""
    afab = schedule_afab.afab_schedule(stage, 4, 8)
    onefonebe = schedule_1f1b.onefonebe_schedule(stage, 4, 8)

    assert len(afab) == len(onefonebe) == 16
    assert sorted(str(op) for op in afab) == sorted(str(op) for op in onefonebe), (
        f"stage {stage}: 1F1B must run exactly the same 8 forwards and 8 backwards "
        "as AFAB, just interleaved differently"
    )


def test_1f1b_last_stage_alternates_immediately():
    """The clearest case, and a good one to check by eye.

    Stage 3 of 4 should be F0 B0 F1 B1 F2 B2 ...
    """
    ops = schedule_1f1b.onefonebe_schedule(stage_index=3, n_stages=4, n_micro=4)
    kinds = "".join(op.kind for op in ops)
    assert kinds == "FBFBFBFB", f"the last stage should alternate from the start; got {kinds}"


def test_1f1b_stage_zero_has_a_long_warmup_and_drain():
    ops = schedule_1f1b.onefonebe_schedule(stage_index=0, n_stages=4, n_micro=8)
    kinds = "".join(op.kind for op in ops)

    assert kinds.startswith("FFFF"), f"stage 0 of 4 should open with 4 forwards; got {kinds}"
    assert kinds.endswith("BBB"), f"stage 0 should drain with backwards at the end; got {kinds}"


def test_1f1b_never_backwards_a_microbatch_it_has_not_forwarded():
    """A schedule that violates this deadlocks. Cheap to check, expensive to debug."""
    for stage in range(4):
        ops = schedule_1f1b.onefonebe_schedule(stage, 4, 8)
        forwarded = set()
        for op in ops:
            if op.kind == "F":
                forwarded.add(op.micro)
            else:
                assert op.micro in forwarded, (
                    f"stage {stage}: backward for micro-batch {op.micro} is scheduled "
                    f"before its forward. Sequence: {''.join(str(o) for o in ops)}"
                )


# --------------------------------------------------------------------------- #
# the memory claim, as pure schedule analysis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_micro", [4, 8, 16])
def test_afab_holds_every_microbatch(n_micro):
    ops = schedule_afab.afab_schedule(0, 4, n_micro)
    assert schedule_afab.max_inflight(ops) == n_micro, (
        f"AFAB runs all {n_micro} forwards before any backward, so peak in-flight "
        f"activations is {n_micro}"
    )


@pytest.mark.parametrize("n_micro", [8, 16, 32])
def test_1f1b_bounds_inflight_by_stage_depth(n_micro):
    """The headline result, verified without allocating anything.

    AFAB holds `m` micro-batches. 1F1B holds about `p`. At `m=32, p=4` that is an
    eightfold difference in activation memory, for the same bubble and the same
    arithmetic.
    """
    n_stages = 4
    ops = schedule_1f1b.onefonebe_schedule(0, n_stages, n_micro)
    peak = schedule_afab.max_inflight(ops)

    assert peak <= n_stages, (
        f"1F1B at p={n_stages}, m={n_micro} held {peak} micro-batches in flight. "
        f"It should be bounded by p={n_stages}, not by m. A peak equal to m means "
        "the schedule is still AFAB."
    )
    assert peak < n_micro, "1F1B must hold strictly fewer activations than AFAB"


def test_the_memory_gap_widens_with_microbatch_count():
    """AFAB grows with m, 1F1B does not. Plot these two if it helps."""
    ratios = []
    for m in (4, 8, 16, 32):
        afab = schedule_afab.max_inflight(schedule_afab.afab_schedule(0, 4, m))
        one = schedule_afab.max_inflight(schedule_1f1b.onefonebe_schedule(0, 4, m))
        ratios.append(afab / one)

    assert ratios == sorted(ratios), (
        f"the AFAB-to-1F1B memory ratio should grow with m; got {ratios}"
    )


# --------------------------------------------------------------------------- #
# load balance
# --------------------------------------------------------------------------- #


def test_stage_flops_reveals_the_lm_head_imbalance():
    """THEORY section 5: an even layer split is uneven in FLOPs.

    With a small vocabulary the imbalance is mild. With V=128000 the LM head alone
    outweighs a transformer block and the last stage becomes the bottleneck for the
    whole pipeline.
    """
    config = ref.PipeConfig(n_layer=6, n_embd=384, vocab_size=72)
    model = ref.build_model(config)
    stages = partition.split_model(model, 3)

    small_vocab = [partition.stage_flops(s, batch=8, seq=128, hidden=384, vocab=72) for s in stages]
    big_vocab = [
        partition.stage_flops(s, batch=8, seq=128, hidden=384, vocab=128_000) for s in stages
    ]

    print(f"\n  FLOPs per stage, V=72:      {small_vocab}")
    print(f"  FLOPs per stage, V=128000:  {big_vocab}")

    small_imbalance = max(small_vocab) / min(small_vocab)
    big_imbalance = max(big_vocab) / min(big_vocab)

    assert big_imbalance > small_imbalance, (
        f"imbalance went from {small_imbalance:.2f}x to {big_imbalance:.2f}x when the "
        "vocabulary grew. It should get worse, not better: the LM head lives on one "
        "stage and scales with V."
    )
    assert big_vocab[-1] == max(big_vocab), (
        "with a large vocabulary the last stage, which owns the LM head, should be the slowest"
    )
