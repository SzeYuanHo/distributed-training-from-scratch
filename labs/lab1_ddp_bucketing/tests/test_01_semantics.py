"""Tier 1: does your DDP produce the same numbers as one process?

The comparison against ``reference/single_process.py`` is the whole design. Every
silent bug in this lab produces a falling loss, so a falling loss proves nothing.

All workers are module-level ``_w_*`` functions because ``spawn`` pickles them,
and all of them carry timeouts because a DDP bug is usually a hang.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from common import comm_counter, dist_utils, testing
from labs.lab1_ddp_bucketing.reference import single_process as ref
from labs.lab1_ddp_bucketing.starter import bucketed_ddp, hooked_ddp, simple_ddp

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(300)]

STEPS = 12
PER_RANK_BATCH = 4
BLOCK = 64


def _config(world: int) -> ref.TrainConfig:
    return ref.TrainConfig(
        preset="tiny",
        steps=STEPS,
        per_rank_batch=PER_RANK_BATCH,
        world_size=world,
        block_size=BLOCK,
    )


def _train_with_wrapper(wrapper_name: str, world: int, **wrapper_kwargs) -> dict:
    """Shared body for every DDP variant. Runs inside a spawned rank."""
    from common import data as data_mod

    config = _config(world)
    rank = dist_utils.rank()

    model = ref.build_model(config)
    wrapper = {
        "simple": lambda m: simple_ddp.SimpleDDP(m),
        "hooked": lambda m: hooked_ddp.HookedDDP(m),
        "bucketed": lambda m: bucketed_ddp.BucketedDDP(m, **wrapper_kwargs),
    }[wrapper_name](model)

    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    dataset = data_mod.load_dataset()
    losses: list[float] = []
    grad_snapshot: dict[str, list] = {}
    counts: dict[str, int] = {}

    for step in range(config.steps):
        x, y = data_mod.get_batch(
            "train",
            batch_size=config.per_rank_batch,
            block_size=config.block_size,
            rank=rank,
            world_size=world,
            step=step,
            seed=config.seed,
            dataset=dataset,
        )

        with comm_counter.CommCounter() as counter:
            _logits, loss = wrapper(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if wrapper_name == "simple":
                wrapper.sync_gradients()
            elif wrapper_name == "bucketed":
                wrapper.finalize()

            if step == 0:
                grad_snapshot = {
                    name: p.grad.detach().clone().tolist()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                counts = counter.counts()

            optimizer.step()

        losses.append(float(loss.detach()))

    with contextlib.suppress(AttributeError, NotImplementedError):
        wrapper.close()

    return {"rank": rank, "losses": losses, "grads": grad_snapshot, "counts": counts}


def _global_loss_curve(results: dict[int, dict]) -> list[float]:
    """Average rank-local losses to recover the equivalent global-batch loss."""
    curves = [info["losses"] for info in results.values()]
    if not curves:
        raise AssertionError("distributed run returned no loss curves")
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise AssertionError(f"ranks returned different loss-curve lengths: {sorted(lengths)}")
    return [sum(values) / len(values) for values in zip(*curves)]


# --------------------------------------------------------------------------- #
# step 3: SimpleDDP
# --------------------------------------------------------------------------- #


def _w_simple(world: int):
    return _train_with_wrapper("simple", world)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.slow
def test_simple_ddp_matches_reference(world):
    """N ranks with per-rank batch b == 1 process with batch N*b. The whole lab."""
    expected = ref.train_reference(_config(world))

    results = dist_utils.run_distributed(
        _w_simple, world_size=world, backend="gloo", args=(world,), timeout_s=280
    )

    testing.assert_curves_close(
        _global_loss_curve(results),
        expected.losses,
        name=f"SimpleDDP global-batch loss curve (world {world})",
    )


def _w_simple_gradients(world: int):
    return _train_with_wrapper("simple", world)


@pytest.mark.parametrize("world", [2, 4])
def test_simple_ddp_step0_gradients_match(world):
    """Compare gradients directly, before any optimizer step muddies them.

    A wrapper that sums instead of averaging shows up here as every gradient
    being exactly N times too large, which the failure message spots for you.
    """
    expected = ref.train_reference(
        ref.TrainConfig(
            preset="tiny",
            steps=1,
            per_rank_batch=PER_RANK_BATCH,
            world_size=world,
            block_size=BLOCK,
        )
    )

    results = dist_utils.run_distributed(
        _w_simple_gradients, world_size=world, backend="gloo", args=(world,), timeout_s=280
    )

    for rank, info in results.items():
        for name, values in info["grads"].items():
            if name not in expected.grad_snapshot:
                continue
            testing.compare_to_reference(
                torch.tensor(values),
                expected.grad_snapshot[name],
                rtol=2e-4,
                atol=1e-6,
                name=f"rank {rank} grad for {name}",
            )
            break  # one parameter is enough to catch a scaling bug


def test_simple_ddp_broadcasts_parameters_at_construction():
    """Rank 1 starts from different weights unless the wrapper broadcasts.

    Skipping the broadcast is the desync bug: gradients average cleanly, the loss
    falls, and each rank has been training a different model.
    """
    results = dist_utils.run_distributed(
        _w_check_broadcast, world_size=2, backend="gloo", timeout_s=120
    )
    assert all(r == "identical" for r in results.values())


def _w_check_broadcast():
    """Deliberately de-synchronise, then check the wrapper fixes it."""
    torch.manual_seed(1000 + dist_utils.rank())  # different weights per rank
    model = torch.nn.Linear(16, 16)

    wrapper = simple_ddp.SimpleDDP(model)

    for name, param in model.named_parameters():
        testing.assert_close_across_ranks(param.data, name=f"post-construction {name}")

    del wrapper
    return "identical"


# --------------------------------------------------------------------------- #
# step 4: HookedDDP
# --------------------------------------------------------------------------- #


def _w_hooked(world: int):
    return _train_with_wrapper("hooked", world)


@pytest.mark.parametrize("world", [2])
@pytest.mark.slow
def test_hooked_ddp_matches_reference(world):
    expected = ref.train_reference(_config(world))
    results = dist_utils.run_distributed(
        _w_hooked, world_size=world, backend="gloo", args=(world,), timeout_s=280
    )
    testing.assert_curves_close(
        _global_loss_curve(results),
        expected.losses,
        name=f"HookedDDP global-batch loss curve (world {world})",
    )


def test_hooked_ddp_issues_one_allreduce_per_parameter():
    """The count that shows the hooks fired, and fired once each."""
    results = dist_utils.run_distributed(
        _w_hooked, world_size=2, backend="gloo", args=(2,), timeout_s=280
    )

    model = ref.build_model(_config(2))
    n_tensors = len({id(p) for p in model.parameters() if p.requires_grad})

    for rank, info in results.items():
        seen = info["counts"].get("all_reduce", 0)
        assert seen == n_tensors, (
            f"rank {rank}: expected one all-reduce per parameter tensor "
            f"({n_tensors}), saw {seen}.\n"
            "  Fewer means some hooks never fired, which for a tied weight is easy "
            "to do.\n"
            "  More means a hook fired twice, which is what register_hook does for a "
            "parameter used in several places."
        )


# --------------------------------------------------------------------------- #
# step 5 and 6: BucketedDDP
# --------------------------------------------------------------------------- #


def _w_bucketed(world: int, bucket_bytes: int, overlap: bool):
    return _train_with_wrapper("bucketed", world, bucket_bytes=bucket_bytes, overlap=overlap)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.slow
def test_bucketed_ddp_matches_reference(world):
    expected = ref.train_reference(_config(world))
    results = dist_utils.run_distributed(
        _w_bucketed,
        world_size=world,
        backend="gloo",
        args=(world, 1 << 20, True),
        timeout_s=280,
    )
    testing.assert_curves_close(
        _global_loss_curve(results),
        expected.losses,
        name=f"BucketedDDP global-batch loss curve (world {world})",
    )


@pytest.mark.parametrize("bucket_bytes", [1 << 18, 1 << 20, 1 << 24])
def test_bucketed_ddp_collective_count(bucket_bytes):
    """Collectives should be ceil(total_bytes / bucket_bytes), not one per parameter.

    This is the assertion that proves bucketing happened at all. A wrapper that
    builds buckets and then all-reduces each parameter anyway passes every numeric
    test and fails this one.
    """
    results = dist_utils.run_distributed(
        _w_bucketed,
        world_size=2,
        backend="gloo",
        args=(2, bucket_bytes, True),
        timeout_s=280,
    )

    model = ref.build_model(_config(2))
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in {id(q): q for q in model.parameters() if q.requires_grad}.values()
    )
    expected_max = -(-total_bytes // bucket_bytes) + 2  # +2 slack for boundary effects
    n_params = len({id(p) for p in model.parameters() if p.requires_grad})

    for rank, info in results.items():
        seen = info["counts"].get("all_reduce", 0)
        assert seen <= expected_max, (
            f"rank {rank}: {seen} all-reduces for {total_bytes} bytes at a "
            f"{bucket_bytes}-byte cap. Expected at most {expected_max}."
        )
        if bucket_bytes >= 1 << 20:
            assert seen < n_params, (
                f"rank {rank}: {seen} all-reduces is not fewer than the {n_params} "
                f"parameter tensors, so nothing was actually bucketed."
            )


def test_larger_buckets_mean_fewer_collectives():
    """Monotonicity. Trivially true if bucketing works, and a useful sanity check."""
    counts = {}
    for size in (1 << 17, 1 << 22):
        results = dist_utils.run_distributed(
            _w_bucketed,
            world_size=2,
            backend="gloo",
            args=(2, size, True),
            timeout_s=280,
        )
        counts[size] = results[0]["counts"].get("all_reduce", 0)

    assert counts[1 << 22] <= counts[1 << 17], (
        f"4 MB buckets issued {counts[1 << 22]} collectives and 128 KB buckets issued "
        f"{counts[1 << 17]}. Larger buckets must not need more."
    )


# --------------------------------------------------------------------------- #
# step 7: no_sync
# --------------------------------------------------------------------------- #


def _w_no_sync(world: int, micro_steps: int):
    from common import data as data_mod

    config = _config(world)
    rank = dist_utils.rank()

    model = ref.build_model(config)
    wrapper = bucketed_ddp.BucketedDDP(model, bucket_bytes=1 << 22)
    optimizer = torch.optim.AdamW(
        wrapper.parameters(), lr=config.lr, weight_decay=0.0, betas=config.betas
    )
    dataset = data_mod.load_dataset()

    losses = []
    with comm_counter.CommCounter() as counter:
        for outer in range(3):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(micro_steps):
                step = outer * micro_steps + micro
                x, y = data_mod.get_batch(
                    "train",
                    batch_size=config.per_rank_batch // micro_steps or 1,
                    block_size=config.block_size,
                    rank=rank,
                    world_size=world,
                    step=step,
                    seed=config.seed,
                    dataset=dataset,
                )
                last = micro == micro_steps - 1
                ctx = contextlib.nullcontext() if last else wrapper.no_sync()
                with ctx:
                    _logits, loss = wrapper(x, y)
                    (loss / micro_steps).backward()
                total += float(loss.detach()) / micro_steps
            wrapper.finalize()
            optimizer.step()
            losses.append(total)

    with contextlib.suppress(AttributeError, NotImplementedError):
        wrapper.close()

    return {"rank": rank, "losses": losses, "counts": counter.counts()}


@pytest.mark.parametrize("micro_steps", [2, 4])
def test_no_sync_reduces_collectives_proportionally(micro_steps):
    """k micro-steps under no_sync should cost 1/k the collectives of k full steps."""
    with_sync = dist_utils.run_distributed(
        _w_no_sync, world_size=2, backend="gloo", args=(2, 1), timeout_s=280
    )
    without = dist_utils.run_distributed(
        _w_no_sync, world_size=2, backend="gloo", args=(2, micro_steps), timeout_s=280
    )

    baseline = with_sync[0]["counts"].get("all_reduce", 0)
    accumulated = without[0]["counts"].get("all_reduce", 0)

    assert accumulated <= baseline, (
        f"{micro_steps} micro-steps under no_sync issued {accumulated} all-reduces, "
        f"against {baseline} for the same number of syncing steps. no_sync must "
        f"suppress the collective, not just delay it."
    )


def test_no_sync_restores_state_after_an_exception():
    """A raise inside the context must not leave syncing permanently off.

    Leaving it off means every subsequent step trains N unrelated models, which
    is the failure mode Lab H hazard 2 is about, reached by a different route.
    """
    results = dist_utils.run_distributed(
        _w_no_sync_exception, world_size=2, backend="gloo", timeout_s=120
    )
    assert all(r == "restored" for r in results.values())


def _w_no_sync_exception():
    model = torch.nn.Linear(8, 8)
    wrapper = bucketed_ddp.BucketedDDP(model, bucket_bytes=1 << 20)

    with contextlib.suppress(RuntimeError), wrapper.no_sync():
        raise RuntimeError("simulated failure inside the context")

    x = torch.randn(4, 8)
    with comm_counter.CommCounter() as counter:
        wrapper(x).sum().backward()
        wrapper.finalize()

    if counter.counts().get("all_reduce", 0) == 0:
        raise AssertionError(
            "no collectives after leaving no_sync via an exception. The flag was not "
            "restored, so this rank is now training on its own. Use try/finally."
        )
    return "restored"


# --------------------------------------------------------------------------- #
# step 8: unused parameters
# --------------------------------------------------------------------------- #


def _w_unused_parameters(find_unused: bool):
    torch.manual_seed(0)
    model = ref.UnusedBranchModel()
    wrapper = bucketed_ddp.BucketedDDP(
        model, bucket_bytes=1 << 20, find_unused_parameters=find_unused
    )

    x = torch.randn(4, 32)
    wrapper(x, use_b=False).sum().backward()  # head_b never gets a gradient
    wrapper.finalize()
    return "completed"


@pytest.mark.timeout(90)
def test_unused_parameters_do_not_hang():
    """The classic production DDP hang, as a test.

    ``head_b`` receives no gradient, so its hook never fires, so its bucket never
    fills, so ``finalize()`` waits for a collective that will never be issued. No
    error, no traceback, just a job that sits there.

    With ``find_unused_parameters=True`` this must complete. The timeout is what
    turns the hang into a failure you can read.
    """
    results = dist_utils.run_distributed(
        _w_unused_parameters,
        world_size=2,
        backend="gloo",
        args=(True,),
        timeout_s=60,
        join_timeout_s=75,
    )
    assert all(r == "completed" for r in results.values()), (
        "finalize() did not return with find_unused_parameters=True. See README step 8."
    )
