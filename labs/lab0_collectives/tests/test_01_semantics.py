"""Tier 1: the collectives, on real processes, on gloo.

Every test here spawns ranks. Two consequences you have to design around, and
both are properties of ``spawn`` rather than of this repo:

1. **Worker functions live at module level.** ``spawn`` pickles the target, and a
   function defined inside a test body will not pickle. Every worker below is a
   module-level ``_w_*`` function, and the test is a thin wrapper that calls
   ``run_distributed`` on it.
2. **Assertions run in the child.** A failed assert in a worker surfaces as a
   ``ProcessRaisedException`` carrying the child's traceback. That is legible, but
   it does mean the pytest assertion-rewriting magic is not available, so worker
   asserts spell out their own messages.

Everything is under a timeout. A deadlock has to fail the run rather than stall
it, and this is where deadlocks actually happen.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch

from common import collectives, comm_counter, dist_utils
from labs.lab0_collectives.reference import torch_collectives
from labs.lab0_collectives.starter import allreduce_algos, p2p_collectives

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(180)]

WORLD_SIZES = [2, 3, 4]
POWER_OF_TWO_SIZES = [2, 4, 8]


def _payload(rank: int, length: int = 64, dtype=torch.float32) -> torch.Tensor:
    """Rank-dependent data. Constant payloads hide index bugs, so never use them."""
    generator = torch.Generator().manual_seed(1000 + rank)
    return torch.randn(length, generator=generator, dtype=dtype) + rank


def _expected_total(world: int, length: int = 64, dtype=torch.float32) -> torch.Tensor:
    total = _payload(0, length, dtype).clone()
    for r in range(1, world):
        total += _payload(r, length, dtype)
    return total


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_collective_implementations_use_the_instrumented_wrappers():
    """Reject direct torch.distributed shortcuts that bypass the lab's counter."""
    offenders: list[str] = []
    for module in (p2p_collectives, allreduce_algos):
        tree = ast.parse(inspect.getsource(module), filename=inspect.getsourcefile(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{module.__name__}:{node.lineno} imports {alias.name}"
                    for alias in node.names
                    if alias.name == "torch.distributed"
                    or alias.name.startswith("torch.distributed.")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "torch" and any(
                    alias.name == "distributed" for alias in node.names
                ):
                    offenders.append(f"{module.__name__}:{node.lineno} imports torch.distributed")
                elif node.module and (
                    node.module == "torch.distributed"
                    or node.module.startswith("torch.distributed.")
                ):
                    offenders.append(f"{module.__name__}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Attribute):
                dotted = _dotted_name(node)
                if dotted == "torch.distributed" or dotted.startswith("torch.distributed."):
                    offenders.append(f"{module.__name__}:{node.lineno} uses {dotted}")

    assert not offenders, (
        "Lab 0 implementations must call common.collectives so communication is "
        "observable and the point-to-point constraint is testable:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


# --------------------------------------------------------------------------- #
# step 2: the harness itself
# --------------------------------------------------------------------------- #


def _w_identity():
    return {
        "rank": dist_utils.rank(),
        "world": dist_utils.world_size(),
        "backend": torch.distributed.get_backend(),
    }


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_world_size_and_ranks(world):
    """Nothing to implement. If this fails, run `python tasks.py doctor`."""
    results = dist_utils.run_distributed(_w_identity, world_size=world, backend="gloo")

    assert len(results) == world, f"expected {world} results, got {len(results)}"
    assert sorted(r["rank"] for r in results.values()) == list(range(world))
    assert all(r["world"] == world for r in results.values())
    assert all(r["backend"] == "gloo" for r in results.values())


# --------------------------------------------------------------------------- #
# step 3: broadcast
# --------------------------------------------------------------------------- #


def _w_broadcast(variant: str, src: int):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    tensor = _payload(rank, 32).clone()
    expected = _payload(src, 32)

    fn = p2p_collectives.broadcast_linear if variant == "linear" else p2p_collectives.broadcast_tree

    with comm_counter.CommCounter() as counter:
        fn(tensor, src=src)

    if not torch.allclose(tensor, expected, rtol=1e-6, atol=1e-8):
        raise AssertionError(
            f"rank {rank}: broadcast_{variant} left {tensor[:4].tolist()} "
            f"but rank {src} holds {expected[:4].tolist()}"
        )

    ops = counter.counts()
    collective_ops = {
        k: v
        for k, v in ops.items()
        if k in ("all_reduce", "all_gather", "broadcast", "reduce", "reduce_scatter", "all_to_all")
    }
    if collective_ops:
        raise AssertionError(
            f"rank {rank}: broadcast_{variant} used built-in collectives {collective_ops}. "
            "This lab is about building them from send and recv."
        )

    return {"rank": rank, "world": world, "ops": ops}


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_broadcast_linear(world):
    results = dist_utils.run_distributed(
        _w_broadcast, world_size=world, backend="gloo", args=("linear", 0)
    )
    root = results[0]["ops"]
    assert root.get("send", 0) == world - 1, (
        f"rank 0 should issue {world - 1} sends in a linear broadcast, saw {root.get('send', 0)}"
    )
    for r in range(1, world):
        assert results[r]["ops"].get("recv", 0) == 1, (
            f"rank {r} should receive exactly once, saw {results[r]['ops'].get('recv', 0)}"
        )


@pytest.mark.parametrize("world", [2, 4, 8])
def test_broadcast_tree_is_correct_and_shallower(world):
    results = dist_utils.run_distributed(
        _w_broadcast, world_size=world, backend="gloo", args=("tree", 0)
    )

    total_sends = sum(r["ops"].get("send", 0) for r in results.values())
    assert total_sends == world - 1, (
        f"a tree broadcast delivers to each of the other {world - 1} ranks exactly "
        f"once, so the total send count should be {world - 1}; saw {total_sends}"
    )

    max_sends = max(r["ops"].get("send", 0) for r in results.values())
    expected_depth = (world - 1).bit_length()
    assert max_sends <= expected_depth, (
        f"no rank should send more than ceil(log2({world})) = {expected_depth} times "
        f"in a tree broadcast; the busiest rank sent {max_sends}. "
        "A count of N-1 on one rank means this is still the linear algorithm."
    )


@pytest.mark.parametrize("world", [3, 4])
def test_broadcast_from_a_nonzero_source(world):
    """src=0 is the easy case. Off-by-one peer arithmetic hides until src != 0."""
    dist_utils.run_distributed(
        _w_broadcast, world_size=world, backend="gloo", args=("tree", world - 1)
    )


# --------------------------------------------------------------------------- #
# step 3c: reduce
# --------------------------------------------------------------------------- #


def _w_reduce_to_root():
    rank, world = dist_utils.rank(), dist_utils.world_size()
    tensor = _payload(rank, 48).clone()
    p2p_collectives.reduce_to_root(tensor, dst=0)

    if rank == 0:
        expected = _expected_total(world, 48)
        if not torch.allclose(tensor, expected, rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"rank 0 holds {tensor[:4].tolist()}, expected {expected[:4].tolist()}"
            )
    return rank


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_reduce_to_root(world):
    dist_utils.run_distributed(_w_reduce_to_root, world_size=world, backend="gloo")


# --------------------------------------------------------------------------- #
# step 4: the two ring phases
# --------------------------------------------------------------------------- #


def _w_reduce_scatter_ring(length: int):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    tensor = _payload(rank, length).clone()
    total = _expected_total(world, length)

    with comm_counter.CommCounter() as counter:
        chunk = p2p_collectives.reduce_scatter_ring(tensor)

    # Every rank works out its own chunk boundaries the same way.
    base, remainder = divmod(length, world)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    expected = total[start:stop]

    if chunk.shape != expected.shape:
        raise AssertionError(
            f"rank {rank}: chunk shape {tuple(chunk.shape)} != {tuple(expected.shape)}. "
            f"With length {length} on {world} ranks the chunks are not all equal."
        )
    if not torch.allclose(chunk, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError(
            f"rank {rank}: reduce-scatter chunk wrong.\n"
            f"  got      {chunk[:4].tolist()}\n"
            f"  expected {expected[:4].tolist()}"
        )

    return {"rank": rank, "ops": counter.counts()}


@pytest.mark.parametrize("world,length", [(2, 64), (3, 64), (4, 64), (3, 7), (4, 100)])
def test_reduce_scatter_ring(world, length):
    results = dist_utils.run_distributed(
        _w_reduce_scatter_ring, world_size=world, backend="gloo", args=(length,)
    )
    for rank, info in results.items():
        sends = info["ops"].get("send", 0) + info["ops"].get("isend", 0)
        assert sends == world - 1, (
            f"rank {rank}: a ring reduce-scatter is exactly {world - 1} sends per rank, "
            f"saw {sends}. More than that means an extra pass over the data."
        )


def _w_all_gather_ring(length: int):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    base, remainder = divmod(length, world)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)

    full = torch.arange(length, dtype=torch.float32)
    chunk = full[start:stop].clone()

    with comm_counter.CommCounter() as counter:
        gathered = p2p_collectives.all_gather_ring(chunk, length)

    if not torch.equal(gathered, full):
        raise AssertionError(
            f"rank {rank}: all-gather produced {gathered[:8].tolist()}, "
            f"expected {full[:8].tolist()}"
        )
    return {"rank": rank, "ops": counter.counts()}


@pytest.mark.parametrize("world,length", [(2, 64), (3, 7), (4, 100)])
def test_all_gather_ring(world, length):
    results = dist_utils.run_distributed(
        _w_all_gather_ring, world_size=world, backend="gloo", args=(length,)
    )
    for rank, info in results.items():
        sends = info["ops"].get("send", 0) + info["ops"].get("isend", 0)
        assert sends == world - 1, f"rank {rank}: expected {world - 1} sends, saw {sends}"


# --------------------------------------------------------------------------- #
# step 5: ring all-reduce
# --------------------------------------------------------------------------- #


def _w_all_reduce(variant: str, length: int):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    tensor = _payload(rank, length).clone()
    reference = torch_collectives.all_reduce(_payload(rank, length).clone())

    fn = {
        "ring": allreduce_algos.all_reduce_ring,
        "naive": allreduce_algos.all_reduce_naive,
        "recursive_doubling": allreduce_algos.all_reduce_recursive_doubling,
    }[variant]

    with comm_counter.CommCounter() as counter:
        fn(tensor)

    if not torch.allclose(tensor, reference, rtol=1e-5, atol=1e-6):
        worst = int((tensor - reference).abs().argmax())
        raise AssertionError(
            f"rank {rank}: all_reduce_{variant} disagrees with dist.all_reduce.\n"
            f"  worst element {worst}: got {tensor[worst]:.8f}, "
            f"expected {reference[worst]:.8f}\n"
            f"  max abs diff {(tensor - reference).abs().max():.3e}"
        )

    return {"rank": rank, "world": world, "ops": counter.counts(), "bytes": counter.total_bytes}


@pytest.mark.parametrize("world", WORLD_SIZES)
@pytest.mark.parametrize("length", [64, 7, 1000])
def test_all_reduce_ring(world, length):
    results = dist_utils.run_distributed(
        _w_all_reduce, world_size=world, backend="gloo", args=("ring", length)
    )

    for rank, info in results.items():
        sends = info["ops"].get("send", 0) + info["ops"].get("isend", 0)
        assert sends == 2 * (world - 1), (
            f"rank {rank}: a ring all-reduce is exactly 2(N-1) = {2 * (world - 1)} sends "
            f"per rank, saw {sends}.\n"
            "  Fewer means a phase is missing. More means a redundant pass that moves "
            "real bytes even though the answer comes out right, which is exactly the "
            "class of bug a numeric test cannot see."
        )


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_all_reduce_naive(world):
    dist_utils.run_distributed(_w_all_reduce, world_size=world, backend="gloo", args=("naive", 64))


@pytest.mark.parametrize("world", POWER_OF_TWO_SIZES)
def test_all_reduce_recursive_doubling(world):
    results = dist_utils.run_distributed(
        _w_all_reduce,
        world_size=world,
        backend="gloo",
        args=("recursive_doubling", 256),
    )

    import math

    for rank, info in results.items():
        sends = info["ops"].get("send", 0) + info["ops"].get("isend", 0)
        rounds = int(math.log2(world))
        assert sends in (rounds, 2 * rounds), (
            f"rank {rank}: expected log2(N)={rounds} sends (whole-buffer variant) or "
            f"2*log2(N)={2 * rounds} (halving-doubling variant), saw {sends}"
        )


def _w_recursive_doubling_rejects_odd():
    tensor = _payload(dist_utils.rank(), 32).clone()
    try:
        allreduce_algos.all_reduce_recursive_doubling(tensor)
    except ValueError as exc:
        if str(dist_utils.world_size()) not in str(exc):
            raise AssertionError(f"the ValueError should name the world size; got: {exc}") from exc
        return "raised"
    raise AssertionError(
        "recursive doubling returned a result at a non-power-of-two world size. "
        "A wrong answer here surfaces three labs later as an inexplicable gradient "
        "mismatch; raise instead."
    )


def test_recursive_doubling_rejects_non_power_of_two():
    results = dist_utils.run_distributed(
        _w_recursive_doubling_rejects_odd, world_size=3, backend="gloo"
    )
    assert all(v == "raised" for v in results.values())


# --------------------------------------------------------------------------- #
# step 7: all-to-all
# --------------------------------------------------------------------------- #


def _w_all_to_all(per_rank: int):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    tensor = torch.arange(world * per_rank, dtype=torch.float32) + rank * 100
    reference = torch_collectives.all_to_all(tensor.clone())

    result = p2p_collectives.all_to_all_p2p(tensor.clone())

    if not torch.allclose(result, reference):
        raise AssertionError(
            f"rank {rank}: all-to-all mismatch.\n"
            f"  got      {result.tolist()}\n"
            f"  expected {reference.tolist()}"
        )
    return rank


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_all_to_all(world):
    dist_utils.run_distributed(_w_all_to_all, world_size=world, backend="gloo", args=(3,))


# --------------------------------------------------------------------------- #
# the deadlock demonstration
# --------------------------------------------------------------------------- #


def _w_symmetric_send_first():
    """Every rank sends before any rank receives. This is supposed to hang.

    The message is deliberately large so a transport is less likely to hide the
    unsafe ordering with eager buffering. Buffering and progress rules vary by
    backend and build; correct code cannot depend on a small send completing.
    """
    rank, world = dist_utils.rank(), dist_utils.world_size()
    peer = (rank + 1) % world

    payload = torch.ones(8_000_000)  # 32 MB, well past any eager threshold
    inbox = torch.empty_like(payload)

    collectives.send(payload, dst=peer)  # everyone sends...
    collectives.recv(inbox, src=(rank - 1) % world)  # ...nobody has received yet
    return "did not deadlock"


@pytest.mark.negative
@pytest.mark.timeout(90)
def test_symmetric_send_deadlocks():
    """A demonstration, not a bug hunt. It passes only on a timeout.

    Three ways out of this, and Lab 3 makes you pick one:

    * odd/even ordering, so odd ranks receive before they send
    * non-blocking isend/irecv with an explicit wait
    * ``dist.batch_isend_irecv``, which hands the whole exchange to the backend

    If the exchange completes, this transport did not expose the hazard at 32 MB;
    record that result rather than treating it as a portable threshold. Unrelated
    worker errors must fail rather than masquerade as the expected deadlock.
    """
    with pytest.raises(Exception) as excinfo:
        dist_utils.run_distributed(
            _w_symmetric_send_first,
            world_size=2,
            backend="gloo",
            timeout_s=25,
            join_timeout_s=40,
        )

    error = excinfo.value
    looks_like_timeout = isinstance(error, dist_utils.DistributedTimeout) or (
        "timed out" in str(error).lower()
    )
    assert looks_like_timeout, (
        "the symmetric exchange failed, but not because it timed out. An unrelated "
        f"{type(error).__name__} must not count as the intended deadlock:\n{error}"
    )


# --------------------------------------------------------------------------- #
# cross-check against the Tier-0 implementation
# --------------------------------------------------------------------------- #


def _w_matches_numpy_version(length: int):
    rank = dist_utils.rank()
    tensor = _payload(rank, length).clone()
    allreduce_algos.all_reduce_ring(tensor)
    return {"rank": rank, "result": tensor.numpy().tolist()}


@pytest.mark.parametrize("world,length", [(3, 7), (4, 64)])
def test_distributed_ring_agrees_with_tier0_ring(world, length):
    """Same algorithm, two implementations. They should not disagree.

    When they do, the Tier-0 one is almost always right and the distributed one
    has a peer-index bug, because Tier 0 has no peers to get wrong.
    """
    from labs.lab0_collectives.starter import ring_numpy

    results = dist_utils.run_distributed(
        _w_matches_numpy_version, world_size=world, backend="gloo", args=(length,)
    )

    buffers = [_payload(r, length).numpy().copy() for r in range(world)]
    tier0 = ring_numpy.ring_allreduce_numpy(buffers)

    for rank in range(world):
        np.testing.assert_allclose(
            np.array(results[rank]["result"]),
            tier0[rank],
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                f"rank {rank}: the distributed ring and the Tier-0 ring disagree. "
                "The Tier-0 one has no peer indices to get wrong, so start there."
            ),
        )
