"""Tier 0: the planner, checked against hand-computed values.

Pure arithmetic. The three configurations here are the ones THEORY section 5 asks
you to work out by hand, so a failure means either your planner or your hand
calculation is wrong, and both are worth knowing about.
"""

from __future__ import annotations

import pytest

from common import cluster_specs
from labs.lab4_capstone_2d.starter import planner

pytestmark = pytest.mark.tier0

MODEL = planner.ModelSpec(
    n_params=7_000_000_000,
    n_layers=32,
    hidden=4096,
    n_heads=32,
    seq_len=4096,
    global_batch_tokens=1_048_576,
)

A100_NODE = cluster_specs.PROFILES["8xA100 (NVLink 3.0)"]
T4_PAIR = cluster_specs.PROFILES["2xT4 (PCIe 3.0)"]
TWO_NODES = cluster_specs.PROFILES["2 nodes x 8xA100 (InfiniBand)"]


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #


def test_enumeration_respects_the_world_size():
    configs = planner.enumerate_configs(8)
    assert configs, "no configurations produced for world size 8"
    for config in configs:
        assert config.world_size == 8, f"{config} multiplies to {config.world_size}, not 8"


def test_enumeration_caps_tensor_parallelism():
    """TP past a fast-interconnect domain is dominated by communication.

    Lab 2 derived where that limit falls. Enumerating tp=64 candidates wastes
    space in a table nobody will read past row 20.
    """
    configs = planner.enumerate_configs(64, max_tp=8)
    assert all(c.tp <= 8 for c in configs), (
        f"max tp in the enumeration is {max(c.tp for c in configs)}, cap was 8"
    )


def test_enumeration_requires_enough_microbatches_for_the_pipeline():
    """m < p means the pipeline never reaches a steady state."""
    configs = planner.enumerate_configs(8, micro_batch_options=(1, 2, 4, 8, 16))
    for config in configs:
        if config.pp > 1:
            assert config.micro_batches >= config.pp, (
                f"{config}: {config.micro_batches} micro-batches across "
                f"{config.pp} stages never reaches a steady state"
            )


def test_enumeration_includes_the_obvious_candidates():
    configs = planner.enumerate_configs(8)
    triples = {(c.dp, c.tp, c.pp) for c in configs}

    for expected in [(8, 1, 1), (1, 8, 1), (2, 4, 1), (4, 2, 1), (2, 2, 2)]:
        assert expected in triples, (
            f"dp={expected[0]}, tp={expected[1]}, pp={expected[2]} is a reasonable "
            "configuration and is missing from the enumeration"
        )


# --------------------------------------------------------------------------- #
# scoring, against hand-computed values
# --------------------------------------------------------------------------- #


def test_model_rejects_a_global_batch_that_cuts_through_a_sequence():
    with pytest.raises(ValueError, match="whole number"):
        planner.ModelSpec(
            n_params=7_000_000_000,
            n_layers=32,
            hidden=4096,
            n_heads=32,
            seq_len=4096,
            global_batch_tokens=1_000_000,
        )


def test_micro_batch_shape_uses_data_parallelism_and_accumulation():
    config = planner.Config(dp=4, tp=2, pp=1, micro_batches=8, zero_stage=2)

    sequences, tokens = MODEL.micro_batch_shape(config)

    assert sequences == 8
    assert tokens == 8 * MODEL.seq_len
    assert tokens * config.dp * config.micro_batches == MODEL.global_batch_tokens


def test_nondivisible_micro_batch_shape_is_infeasible():
    twenty_four_sequences = planner.ModelSpec(
        n_params=125_000_000,
        n_layers=12,
        hidden=768,
        n_heads=12,
        seq_len=1024,
        global_batch_tokens=24 * 1024,
    )
    config = planner.Config(dp=4, tp=1, pp=1, micro_batches=8, zero_stage=1)

    scored = planner.score(twenty_four_sequences, config, A100_NODE)

    assert not scored.feasible
    assert any("do not divide evenly" in reason for reason in scored.reasons)
    assert scored.micro_batch_sequences == 0
    assert scored.micro_batch_tokens == 0


def test_ranking_places_a_nondivisible_batch_candidate_last():
    twenty_four_sequences = planner.ModelSpec(
        n_params=125_000_000,
        n_layers=12,
        hidden=768,
        n_heads=12,
        seq_len=1024,
        global_batch_tokens=24 * 1024,
    )
    valid = planner.Config(dp=4, tp=1, pp=1, micro_batches=2, zero_stage=1)
    nondivisible = planner.Config(dp=4, tp=1, pp=1, micro_batches=8, zero_stage=1)

    ranked = planner.rank_configs(
        twenty_four_sequences,
        [nondivisible, valid],
        A100_NODE,
    )

    assert ranked[0].config == valid
    assert ranked[0].feasible
    assert ranked[-1].config == nondivisible
    assert not ranked[-1].feasible


def test_activation_memory_tracks_the_derived_micro_batch_size():
    large_micro = planner.score(
        MODEL,
        planner.Config(dp=4, tp=2, pp=1, micro_batches=4, zero_stage=2),
        A100_NODE,
    )
    small_micro = planner.score(
        MODEL,
        planner.Config(dp=4, tp=2, pp=1, micro_batches=16, zero_stage=2),
        A100_NODE,
    )

    assert large_micro.micro_batch_sequences == 16
    assert small_micro.micro_batch_sequences == 4
    assert large_micro.activation_memory == 4 * small_micro.activation_memory


def test_bf16_is_rejected_on_a_profile_without_bf16_tensor_cores():
    config = planner.Config(dp=1, tp=2, pp=1, micro_batches=8, zero_stage=3)

    scored = planner.score(MODEL, config, T4_PAIR)

    assert not scored.feasible
    assert any("bf16" in reason.lower() and "not supported" in reason for reason in scored.reasons)


def test_pure_data_parallel_does_not_fit_a_7b_model():
    """dp=8, tp=1, pp=1, zero=0: every rank holds the whole 16-byte-per-param state.

    7e9 * 16 = 112 GB per rank, against 80 GB of VRAM. Does not fit, and the
    reason should say so in bytes.
    """
    config = planner.Config(dp=8, tp=1, pp=1, micro_batches=1, zero_stage=0)
    scored = planner.score(MODEL, config, A100_NODE)

    assert not scored.feasible
    assert scored.memory_per_gpu > 100e9, (
        f"7B params at 16 bytes each is 112 GB of state; the planner said "
        f"{scored.memory_per_gpu / 1e9:.1f} GB"
    )
    assert scored.reasons, "an infeasible configuration should say why"


def test_zero3_makes_pure_data_parallel_fit():
    """dp=8, zero=3 shards everything: 112/8 = 14 GB of state per rank."""
    config = planner.Config(dp=8, tp=1, pp=1, micro_batches=1, zero_stage=3)
    scored = planner.score(MODEL, config, A100_NODE)

    assert scored.param_memory < 20e9, (
        f"ZeRO-3 across 8 ranks should leave about 14 GB of state per rank, got "
        f"{scored.param_memory / 1e9:.1f} GB"
    )


def test_tensor_parallelism_also_reduces_parameter_memory():
    """tp=8 shards the parameters across the TP dimension, like ZeRO-3 does across dp."""
    alone = planner.score(
        MODEL,
        planner.Config(dp=8, tp=1, pp=1, micro_batches=1, zero_stage=0),
        A100_NODE,
    )
    sharded = planner.score(
        MODEL,
        planner.Config(dp=1, tp=8, pp=1, micro_batches=1, zero_stage=0),
        A100_NODE,
    )

    assert sharded.param_memory < alone.param_memory / 4, (
        f"tp=8 gave {sharded.param_memory / 1e9:.1f} GB against dp=8's "
        f"{alone.param_memory / 1e9:.1f} GB. Tensor parallelism shards parameters "
        "within a layer, so it should be a comparable reduction to ZeRO-3."
    )


def test_tp_communication_does_not_fall_with_tp_degree():
    """Lab 2's central result, restated in the planner.

    The all-reduced tensor is (b, s, h) whatever tp is. Compute per rank falls as
    1/tp and communication does not, which is the ceiling.
    """
    volumes = {}
    for tp in (2, 4, 8):
        config = planner.Config(dp=8 // tp, tp=tp, pp=1, micro_batches=1, zero_stage=1)
        volumes[tp] = planner.score(MODEL, config, A100_NODE).comm_tp

    assert volumes[8] >= volumes[2] * 0.8, (
        f"TP communication came out {volumes} across tp=2,4,8. It should be roughly "
        "flat: the tensor being reduced is (b, s, h) regardless of tp."
    )


def test_pipeline_bubble_appears_in_the_score():
    no_pipe = planner.score(
        MODEL,
        planner.Config(dp=8, tp=1, pp=1, micro_batches=8, zero_stage=1),
        A100_NODE,
    )
    piped = planner.score(
        MODEL,
        planner.Config(dp=2, tp=1, pp=4, micro_batches=8, zero_stage=1),
        A100_NODE,
    )

    assert no_pipe.bubble_overhead == 0.0, "pp=1 has no bubble"
    assert piped.bubble_overhead == pytest.approx(3 / 8, rel=1e-6), (
        f"pp=4, m=8 should give overhead (4-1)/8 = 0.375, got {piped.bubble_overhead}"
    )


def test_more_microbatches_shrink_the_bubble():
    bubbles = []
    for m in (4, 8, 16, 32):
        config = planner.Config(dp=2, tp=1, pp=4, micro_batches=m, zero_stage=1)
        bubbles.append(planner.score(MODEL, config, A100_NODE).bubble_overhead)

    assert bubbles == sorted(bubbles, reverse=True), (
        f"bubble should fall as micro-batches rise; got {bubbles}"
    )


def test_activation_memory_responds_to_recomputation():
    model_full = planner.ModelSpec(
        n_params=MODEL.n_params,
        n_layers=MODEL.n_layers,
        hidden=MODEL.hidden,
        n_heads=MODEL.n_heads,
        seq_len=MODEL.seq_len,
        global_batch_tokens=MODEL.global_batch_tokens,
        recompute="full",
    )
    config = planner.Config(dp=8, tp=1, pp=1, micro_batches=8, zero_stage=3)

    without = planner.score(MODEL, config, A100_NODE).activation_memory
    with_recompute = planner.score(model_full, config, A100_NODE).activation_memory

    assert with_recompute < without, (
        f"full recomputation gave {with_recompute / 1e9:.1f} GB of activations "
        f"against {without / 1e9:.1f} GB without. It should be much lower."
    )


# --------------------------------------------------------------------------- #
# the ranking, which is the deliverable
# --------------------------------------------------------------------------- #


def test_ranking_puts_feasible_configurations_first():
    configs = planner.enumerate_configs(8)
    scored = planner.rank_configs(MODEL, configs, A100_NODE)

    feasible = [i for i, s in enumerate(scored) if s.feasible]
    infeasible = [i for i, s in enumerate(scored) if not s.feasible]

    if feasible and infeasible:
        assert max(feasible) < min(infeasible), (
            "an infeasible configuration ranked above a feasible one"
        )


def test_ranking_keeps_infeasible_rows():
    """A table that silently drops near misses hides the most useful information."""
    configs = planner.enumerate_configs(8)
    scored = planner.rank_configs(MODEL, configs, A100_NODE)
    assert len(scored) == len(configs), (
        "rank_configs dropped rows. Keep them and sort them last: how close the "
        "near misses came is exactly what you want to know."
    )


def test_the_answer_changes_with_the_interconnect():
    """The whole point. If the ranking is hardware-independent, the model is broken.

    On NVLink at 300 GB/s, tensor parallelism is cheap and a high tp is attractive.
    Across nodes at 25 GB/s it is not, and the planner should notice.
    """
    configs = planner.enumerate_configs(8)

    fast = planner.rank_configs(MODEL, configs, A100_NODE)
    slow = planner.rank_configs(MODEL, configs, TWO_NODES)

    fast_best = fast[0].config
    slow_best = slow[0].config

    print(f"\n  NVLink node   -> {fast_best}")
    print(f"  Across nodes  -> {slow_best}")

    assert (fast_best.tp, fast_best.dp, fast_best.pp) != (
        slow_best.tp,
        slow_best.dp,
        slow_best.pp,
    ) or fast[0].est_step_seconds != slow[0].est_step_seconds, (
        "the planner recommended the same configuration with the same predicted "
        "step time on a 300 GB/s link and a 25 GB/s one. The cost model is not "
        "sensitive to the interconnect, which means it is not doing its job."
    )


def test_small_model_on_t4_pair_is_feasible_somewhere():
    """A sanity check on the other end of the scale."""
    small = planner.ModelSpec(
        n_params=125_000_000,
        n_layers=12,
        hidden=768,
        n_heads=12,
        seq_len=1024,
        global_batch_tokens=131_072,
        dtype="fp16",
    )
    configs = planner.enumerate_configs(2, max_tp=2)
    scored = planner.rank_configs(small, configs, T4_PAIR)

    assert any(s.feasible for s in scored), (
        "no configuration fits a 125M model on two 16 GB T4s, which cannot be right"
    )


def test_report_table_is_readable():
    configs = planner.enumerate_configs(8)
    scored = planner.rank_configs(MODEL, configs, A100_NODE)
    table = planner.report_table(scored, limit=10)

    assert isinstance(table, str) and len(table.splitlines()) >= 5
    lowered = table.lower()
    for token in ("dp", "tp", "pp"):
        assert token in lowered, f"the table should label the {token} column"
    assert "seq/mb" in lowered, "show the derived local micro-batch size in the table"
    assert any(unit in lowered for unit in ("gb", "mb")), (
        "print human units, not raw byte counts. Someone has to read this."
    )
