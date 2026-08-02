"""Tier 1: does the pipeline compute what one process computes?

Every test carries a timeout. The characteristic pipeline failure is a hang, not
an exception, and a hang without a timeout stalls the whole suite.

Compare against ``run_microbatched_reference``, not ``run_reference``.
Micro-batching reorders floating-point additions, so the two differ in the last
digits even when everything is correct, and comparing against the wrong one sends
you hunting for a bug that is arithmetic.
"""

from __future__ import annotations

import pytest

from common import comm_counter, dist_utils, memory
from labs.lab3_pipeline_parallel.reference import single_process as ref
from labs.lab3_pipeline_parallel.starter import partition, schedule_1f1b, schedule_afab

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(180)]


def _make_stage(config: ref.PipeConfig):
    """Every rank builds the same model from the same seed, then keeps its slice."""
    model = ref.build_model(config)
    stages = partition.split_model(model, dist_utils.world_size())
    return stages[dist_utils.rank()], model


def _microbatches(config: ref.PipeConfig, step: int = 0):
    x, y = ref.get_batch(config, step)
    micro = config.micro_batch_size
    xs = [x[i * micro : (i + 1) * micro] for i in range(config.n_micro)]
    ys = [y[i * micro : (i + 1) * micro] for i in range(config.n_micro)]
    return xs, ys


# --------------------------------------------------------------------------- #
# step 2: point-to-point mechanics
# --------------------------------------------------------------------------- #


def _w_two_stage_single_batch():
    config = ref.PipeConfig(n_layer=4, n_micro=1, batch_size=4)
    stage, _model = _make_stage(config)
    xs, ys = _microbatches(config)

    with comm_counter.CommCounter() as counter:
        loss = schedule_afab.run_afab(stage, 1, xs, ys)

    return {"rank": dist_utils.rank(), "loss": float(loss), "counts": counter.counts()}


@pytest.mark.timeout(120)
def test_two_stage_forward_backward():
    """One micro-batch, two stages. Establishes the mechanics before scheduling."""
    config = ref.PipeConfig(n_layer=4, n_micro=1, batch_size=4)
    expected = ref.run_microbatched_reference(config)["losses"][0]

    results = dist_utils.run_distributed(
        _w_two_stage_single_batch,
        world_size=2,
        backend="gloo",
        timeout_s=90,
        join_timeout_s=110,
    )

    last_rank = max(results)
    got = results[last_rank]["loss"]
    assert got == pytest.approx(expected, rel=1e-4), (
        f"the last stage reported loss {got:.6f}, single process gives {expected:.6f}"
    )


def _w_gradients_reach_the_first_stage():
    """Every stage's parameters must receive a gradient.

    The failure this catches: activations sent without ``requires_grad``, so the
    backward pass stops at each stage's input and never crosses back. The later
    stages still learn, the loss still falls, and the early layers are frozen.
    """
    config = ref.PipeConfig(n_layer=4, n_micro=2, batch_size=4)
    stage, _model = _make_stage(config)
    xs, ys = _microbatches(config)

    schedule_afab.run_afab(stage, config.n_micro, xs, ys)

    have_grad = sum(1 for p in stage.parameters() if p.grad is not None)
    total = sum(1 for _ in stage.parameters())

    if have_grad == 0:
        raise AssertionError(
            f"rank {dist_utils.rank()}: none of this stage's {total} parameters got a "
            "gradient.\n"
            "  The autograd graph does not cross a process boundary. The receiving "
            "stage has to mark its input as requiring grad, and then send that "
            "input's gradient back upstream."
        )
    return {"rank": dist_utils.rank(), "with_grad": have_grad, "total": total}


@pytest.mark.parametrize("world", [2, 4])
def test_gradients_reach_every_stage(world):
    results = dist_utils.run_distributed(
        _w_gradients_reach_the_first_stage,
        world_size=world,
        backend="gloo",
        timeout_s=120,
        join_timeout_s=140,
    )
    for rank, info in results.items():
        assert info["with_grad"] == info["total"], (
            f"rank {rank}: only {info['with_grad']} of {info['total']} parameters "
            "received a gradient"
        )


# --------------------------------------------------------------------------- #
# steps 3 and 5: the schedules
# --------------------------------------------------------------------------- #


def _w_run_schedule(kind: str, n_micro: int, n_layer: int):
    from common import timers

    config = ref.PipeConfig(n_layer=n_layer, n_micro=n_micro, batch_size=8)
    stage, _model = _make_stage(config)
    xs, ys = _microbatches(config)

    log = timers.EventLog()
    measurement_device = next(stage.parameters()).device
    memory.reset_peak(measurement_device)

    runner = schedule_afab.run_afab if kind == "afab" else schedule_1f1b.run_1f1b
    with comm_counter.CommCounter() as counter:
        loss = runner(stage, n_micro, xs, ys, event_log=log)

    peak = memory.measured_peak(measurement_device)
    merged = log.gather()

    return {
        "rank": dist_utils.rank(),
        "loss": float(loss),
        "counts": counter.counts(),
        "peak": peak["allocated"],
        "occupancy": log.occupancy(merged),
    }


@pytest.mark.parametrize("world,n_micro", [(2, 4), (4, 8)])
@pytest.mark.slow
def test_afab_matches_reference(world, n_micro):
    config = ref.PipeConfig(n_layer=4, n_micro=n_micro, batch_size=8)
    expected = ref.run_microbatched_reference(config)["losses"][0]

    results = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("afab", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )

    got = results[max(results)]["loss"]
    assert got == pytest.approx(expected, rel=1e-4), (
        f"AFAB with p={world}, m={n_micro} gave loss {got:.6f}, single-process "
        f"micro-batched reference gives {expected:.6f}.\n"
        "  A factor of m usually means the per-micro-batch loss was not scaled by 1/m."
    )


@pytest.mark.parametrize("world,n_micro", [(2, 4), (4, 8)])
@pytest.mark.slow
def test_1f1b_matches_afab(world, n_micro):
    """Same work, different order, same answer."""
    afab = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("afab", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )
    onefonebe = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("1f1b", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )

    last = max(afab)
    assert onefonebe[last]["loss"] == pytest.approx(afab[last]["loss"], rel=1e-5), (
        f"1F1B gave {onefonebe[last]['loss']:.6f} and AFAB gave "
        f"{afab[last]['loss']:.6f}. These are the same arithmetic in a different "
        "order and should agree to floating-point noise."
    )


@pytest.mark.parametrize("world,n_micro", [(2, 8), (4, 8)])
@pytest.mark.slow
def test_peak_activation_memory_is_lower_under_1f1b(world, n_micro):
    """The reason 1F1B is the production default.

    The measured ratio will be well below the schedule-level `m/p`, because the
    measurement includes parameters, the interpreter, and torch itself, none of
    which change between schedules. The direction is what matters, and
    ``test_00_math.py`` checks the exact bound on the schedule itself.
    """
    afab = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("afab", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )
    onefonebe = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("1f1b", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )

    afab_peak = max(r["peak"] for r in afab.values())
    one_peak = max(r["peak"] for r in onefonebe.values())

    print(f"\n  p={world}, m={n_micro}")
    print(f"    AFAB peak {afab_peak / 1e6:8.1f} MB")
    print(f"    1F1B peak {one_peak / 1e6:8.1f} MB")
    print(f"    schedule-level bound predicts a ratio near {n_micro / world:.1f}x;")
    print("    the measured ratio is smaller because parameters and the framework")
    print("    are in both numbers.")

    assert one_peak <= afab_peak * 1.05, (
        f"1F1B peaked at {one_peak / 1e6:.1f} MB against AFAB's "
        f"{afab_peak / 1e6:.1f} MB. 1F1B should never use more.\n"
        "  If they are equal, check that the backward pass runs early enough to "
        "release activations, rather than the activation list simply growing."
    )


@pytest.mark.parametrize("world,n_micro", [(2, 8), (4, 8)])
def test_measured_bubble_is_near_theory(world, n_micro):
    """Measured idle fraction against (p-1)/(m+p-1).

    The measurement will be worse than theory, and the writeup asks you why. The
    tolerance here is deliberately loose: the point is to confirm the model has
    the right shape, not to pretend a gloo pipeline hits its analytic bound.
    """
    results = dist_utils.run_distributed(
        _w_run_schedule,
        world_size=world,
        backend="gloo",
        args=("1f1b", n_micro, 4),
        timeout_s=150,
        join_timeout_s=170,
    )

    theory = (world - 1) / (n_micro + world - 1)
    occupancy = results[0]["occupancy"]
    busy = occupancy.get("compute", 0.0)
    measured_bubble = max(0.0, 1.0 - busy)

    print(f"\n  p={world}, m={n_micro}: theory {theory:.3f}, measured {measured_bubble:.3f}")

    assert measured_bubble >= theory * 0.5, (
        f"measured bubble {measured_bubble:.3f} is far below theory {theory:.3f}, "
        "which usually means the event log is not recording every idle stage"
    )


# --------------------------------------------------------------------------- #
# optional step 7: production API availability
# --------------------------------------------------------------------------- #


def test_optional_torch_pipelining_api_smoke():
    """The optional comparison names APIs that exist on the supported Torch line."""
    pipelining = pytest.importorskip("torch.distributed.pipelining")

    assert pipelining.ScheduleGPipe.__name__ == "ScheduleGPipe"
    assert pipelining.Schedule1F1B.__name__ == "Schedule1F1B"
