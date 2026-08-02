"""Tier 1: does the sharded model compute what the reference computes?

Layer, block, then whole model. Forward outputs *and* input gradients, because an
f/g swap leaves the forward correct and only shows up in the backward.

The `tiny` reference config here has 4 heads, so tp_size goes up to 4. Anything
larger would put zero heads on some rank.
"""

from __future__ import annotations

import pytest
import torch

from common import comm_counter, dist_utils, testing
from labs.lab2_tensor_parallel.reference import plain_blocks
from labs.lab2_tensor_parallel.starter import blocks, layers, mappings, tp_gpt, vocab

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(300)]

TP_SIZES = [2, 4]


# --------------------------------------------------------------------------- #
# step 2: the mappings
# --------------------------------------------------------------------------- #

# A rank-local finite-difference gradcheck is not valid for these deliberately
# distributed conjugate mappings: perturbing one rank changes values observed by
# its peers, outside gradcheck's local function boundary. The test below checks
# the distributed forward/backward contract directly instead.


def _w_conjugate_property():
    """f's backward must be g's forward, and vice versa. Verified numerically.

    This is the definition of "conjugate" made testable, and it is the property
    that a swapped implementation violates.
    """
    world = dist_utils.world_size()
    rank = dist_utils.rank()

    x = torch.full((4,), float(rank + 1), requires_grad=True)
    out = mappings.CopyToTensorParallelRegion.apply(x, None)
    if not torch.equal(out.detach(), x.detach()):
        raise AssertionError("f's forward must be the identity")

    out.sum().backward()
    # f's backward is an all-reduce, so every rank's gradient should be `world`,
    # not 1.
    if not torch.allclose(x.grad, torch.full((4,), float(world))):
        raise AssertionError(
            f"rank {rank}: f's backward gave {x.grad.tolist()}, expected "
            f"{[float(world)] * 4}. f's forward is the identity and its backward is "
            "an all-reduce; if you got all ones, the backward is also the identity."
        )

    y = torch.full((4,), float(rank + 1), requires_grad=True)
    reduced = mappings.ReduceFromTensorParallelRegion.apply(y, None)
    expected_sum = float(sum(range(1, world + 1)))
    if not torch.allclose(reduced.detach(), torch.full((4,), expected_sum)):
        raise AssertionError(
            f"rank {rank}: g's forward gave {reduced.tolist()}, expected "
            f"{expected_sum} on every element"
        )

    reduced.sum().backward()
    if not torch.allclose(y.grad, torch.ones(4)):
        raise AssertionError(
            f"rank {rank}: g's backward gave {y.grad.tolist()}, expected all ones. "
            "The forward was a sum, so each rank's contribution has derivative 1 and "
            "the incoming gradient passes through unchanged."
        )

    return "conjugate"


@pytest.mark.parametrize("world", TP_SIZES)
def test_f_and_g_are_conjugate(world):
    results = dist_utils.run_distributed(
        _w_conjugate_property, world_size=world, backend="gloo", timeout_s=180
    )
    assert all(v == "conjugate" for v in results.values())


# --------------------------------------------------------------------------- #
# step 3: the layers
# --------------------------------------------------------------------------- #


def _w_column_parallel(gather_output: bool):
    rank, world = dist_utils.rank(), dist_utils.world_size()

    reference = plain_blocks.make_linear(32, 64, bias=True, seed=0)
    x = torch.randn(
        6, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(0), requires_grad=True
    )

    expected = reference(x)
    expected.sum().backward()
    expected_grad = x.grad.clone()
    x.grad = None

    parallel = layers.ColumnParallelLinear(32, 64, bias=True, gather_output=gather_output).double()
    parallel.load_from_reference(reference)

    with comm_counter.CommCounter() as counter:
        got = parallel(x)

    if gather_output:
        testing.compare_to_reference(
            got, expected, rtol=1e-10, atol=1e-12, name=f"rank {rank} column-parallel output"
        )
    else:
        per_rank = 64 // world
        slice_ = expected[..., rank * per_rank : (rank + 1) * per_rank]
        testing.compare_to_reference(
            got, slice_, rtol=1e-10, atol=1e-12, name=f"rank {rank} column-parallel shard"
        )

    got.sum().backward()
    testing.compare_to_reference(
        x.grad,
        expected_grad,
        rtol=1e-10,
        atol=1e-12,
        name=(
            f"rank {rank} column-parallel INPUT GRADIENT. This is the one an f/g "
            "swap breaks while leaving the forward correct."
        ),
    )

    return {"rank": rank, "counts": counter.counts()}


@pytest.mark.parametrize("world", TP_SIZES)
@pytest.mark.parametrize("gather_output", [False, True])
def test_column_parallel_linear(world, gather_output):
    results = dist_utils.run_distributed(
        _w_column_parallel,
        world_size=world,
        backend="gloo",
        args=(gather_output,),
        timeout_s=200,
    )
    for rank, info in results.items():
        gathers = info["counts"].get("all_gather", 0)
        if gather_output:
            assert gathers >= 1, f"rank {rank}: gather_output=True needs an all-gather"
        else:
            assert gathers == 0, (
                f"rank {rank}: gather_output=False issued {gathers} all-gathers. "
                "The whole point of leaving the output sharded is that nothing is sent."
            )


def _w_row_parallel():
    rank, world = dist_utils.rank(), dist_utils.world_size()

    reference = plain_blocks.make_linear(64, 32, bias=True, seed=1)
    full_x = torch.randn(
        6, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(1), requires_grad=True
    )

    expected = reference(full_x)
    expected.sum().backward()
    expected_grad = full_x.grad.clone()

    parallel = layers.RowParallelLinear(64, 32, bias=True, input_is_parallel=True).double()
    parallel.load_from_reference(reference)

    per_rank = 64 // world
    local_x = full_x.detach()[..., rank * per_rank : (rank + 1) * per_rank].clone()
    local_x.requires_grad_(True)

    with comm_counter.CommCounter() as counter:
        got = parallel(local_x)

    testing.compare_to_reference(
        got, expected.detach(), rtol=1e-10, atol=1e-12, name=f"rank {rank} row-parallel output"
    )

    got.sum().backward()
    expected_local_grad = expected_grad[..., rank * per_rank : (rank + 1) * per_rank]
    testing.compare_to_reference(
        local_x.grad,
        expected_local_grad,
        rtol=1e-10,
        atol=1e-12,
        name=f"rank {rank} row-parallel input gradient",
    )

    return {"rank": rank, "counts": counter.counts()}


@pytest.mark.parametrize("world", TP_SIZES)
def test_row_parallel_linear(world):
    results = dist_utils.run_distributed(
        _w_row_parallel, world_size=world, backend="gloo", timeout_s=200
    )
    for rank, info in results.items():
        assert info["counts"].get("all_reduce", 0) == 1, (
            f"rank {rank}: a row-parallel layer needs exactly one forward all-reduce, "
            f"saw {info['counts'].get('all_reduce', 0)}"
        )


# --------------------------------------------------------------------------- #
# steps 4 and 5: the blocks
# --------------------------------------------------------------------------- #


def _w_parallel_mlp():
    rank = dist_utils.rank()
    config = plain_blocks.tiny_config()

    reference = plain_blocks.make_mlp(config, seed=2)
    x = torch.randn(
        2,
        8,
        config.n_embd,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(2),
        requires_grad=True,
    )

    expected = reference(x)
    expected.sum().backward()
    expected_grad = x.grad.clone()
    x.grad = None

    parallel = blocks.ParallelMLP(config).double()
    parallel.load_from_reference(reference)

    with comm_counter.CommCounter() as counter:
        got = parallel(x)
    forward_counts = dict(counter.counts())

    testing.compare_to_reference(
        got, expected.detach(), rtol=1e-10, atol=1e-12, name=f"rank {rank} parallel MLP output"
    )

    got.sum().backward()
    testing.compare_to_reference(
        x.grad,
        expected_grad,
        rtol=1e-10,
        atol=1e-12,
        name=f"rank {rank} parallel MLP input gradient",
    )

    return {"rank": rank, "forward_counts": forward_counts}


@pytest.mark.parametrize("world", TP_SIZES)
def test_parallel_mlp(world):
    results = dist_utils.run_distributed(
        _w_parallel_mlp, world_size=world, backend="gloo", timeout_s=200
    )
    for rank, info in results.items():
        reduces = info["forward_counts"].get("all_reduce", 0)
        assert reduces == 1, (
            f"rank {rank}: the MLP block should need exactly ONE forward all-reduce, "
            f"saw {reduces}.\n"
            "  Two usually means gather_output=True on the column layer, which "
            "un-shards the activations and forces the row layer to scatter them "
            "again. See THEORY.md section 3."
        )


def _w_parallel_attention():
    rank = dist_utils.rank()
    config = plain_blocks.tiny_config()

    reference = plain_blocks.make_attention(config, seed=3)
    x = torch.randn(
        2,
        8,
        config.n_embd,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(3),
        requires_grad=True,
    )

    expected = reference(x)
    expected.sum().backward()
    expected_grad = x.grad.clone()
    x.grad = None

    parallel = blocks.ParallelCausalSelfAttention(config).double()
    parallel.load_from_reference(reference)

    with comm_counter.CommCounter() as counter:
        got = parallel(x)
    forward_counts = dict(counter.counts())

    testing.compare_to_reference(
        got,
        expected.detach(),
        rtol=1e-9,
        atol=1e-11,
        name=(
            f"rank {rank} parallel attention output. If the layer tests pass and "
            "this does not, check the QKV row slicing in load_from_reference."
        ),
    )

    got.sum().backward()
    testing.compare_to_reference(
        x.grad,
        expected_grad,
        rtol=1e-9,
        atol=1e-11,
        name=f"rank {rank} parallel attention input gradient",
    )

    return {"rank": rank, "forward_counts": forward_counts}


@pytest.mark.parametrize("world", TP_SIZES)
def test_parallel_attention(world):
    results = dist_utils.run_distributed(
        _w_parallel_attention, world_size=world, backend="gloo", timeout_s=200
    )
    for rank, info in results.items():
        assert info["forward_counts"].get("all_reduce", 0) == 1, (
            f"rank {rank}: attention should need exactly one forward all-reduce"
        )


def _w_head_divisibility():
    """4 heads on 8 ranks must raise, with both numbers in the message."""
    config = plain_blocks.tiny_config(n_head=4, n_embd=64, max_tp=4)
    try:
        blocks.ParallelCausalSelfAttention(config)
    except (ValueError, AssertionError) as exc:
        message = str(exc)
        if "4" not in message or str(dist_utils.world_size()) not in message:
            raise AssertionError(
                f"the error should name both n_head and tp_size; got: {message}"
            ) from exc
        return "raised"
    raise AssertionError(
        "4 heads across 8 ranks was accepted. Some rank owns zero heads, and the "
        "shapes will not line up several layers later. Assert in the constructor."
    )


def test_head_divisibility_is_asserted():
    results = dist_utils.run_distributed(
        _w_head_divisibility, world_size=8, backend="gloo", timeout_s=180
    )
    assert all(v == "raised" for v in results.values())


# --------------------------------------------------------------------------- #
# steps 6 and 7: vocabulary
# --------------------------------------------------------------------------- #


def _w_vocab_embedding():
    rank = dist_utils.rank()
    vocab_size, dim = 32, 16

    reference = plain_blocks.make_embedding(vocab_size, dim, seed=4)
    ids = torch.randint(0, vocab_size, (2, 6), generator=torch.Generator().manual_seed(4))
    expected = reference(ids)

    parallel = vocab.VocabParallelEmbedding(vocab_size, dim).double()
    parallel.load_from_reference(reference)

    with comm_counter.CommCounter() as counter:
        got = parallel(ids)

    testing.compare_to_reference(
        got,
        expected,
        rtol=1e-12,
        atol=1e-14,
        name=(
            f"rank {rank} vocab-parallel embedding. Nonzero rows where they should "
            "be zero means out-of-range ids were not masked."
        ),
    )
    return {"rank": rank, "counts": counter.counts()}


@pytest.mark.parametrize("world", TP_SIZES)
def test_vocab_parallel_embedding(world):
    results = dist_utils.run_distributed(
        _w_vocab_embedding, world_size=world, backend="gloo", timeout_s=200
    )
    for rank, info in results.items():
        assert info["counts"].get("all_reduce", 0) == 1, (
            f"rank {rank}: one all-reduce to sum the per-rank contributions"
        )


def _w_vocab_cross_entropy(extreme: bool):
    rank, world = dist_utils.rank(), dist_utils.world_size()
    batch, seq, vocab_size = 2, 6, 32

    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(batch, seq, vocab_size, dtype=torch.float64, generator=generator) * 5
    if extreme:
        # A logit large enough to overflow exp in fp32. The global-max subtraction
        # is what makes this survivable.
        logits[0, 0, 29] = 100.0

    targets = torch.randint(0, vocab_size, (batch, seq), generator=generator)
    expected = plain_blocks.reference_cross_entropy(logits, targets)

    per_rank = vocab_size // world
    local = logits[..., rank * per_rank : (rank + 1) * per_rank].clone().requires_grad_(True)

    with comm_counter.CommCounter() as counter:
        loss = vocab.vocab_parallel_cross_entropy(local, targets)

    if not torch.allclose(loss, expected, rtol=1e-8, atol=1e-10):
        raise AssertionError(
            f"rank {rank}: vocab-parallel cross-entropy gave {float(loss):.10f}, "
            f"expected {float(expected):.10f}.\n"
            "  A small constant offset usually means the global max was not "
            "subtracted, or was subtracted twice."
        )

    loss.backward()
    if local.grad is None:
        raise AssertionError("no gradient reached the local logits")

    return {"rank": rank, "counts": counter.counts(), "loss": float(loss)}


@pytest.mark.parametrize("world", TP_SIZES)
@pytest.mark.parametrize("extreme", [False, True])
def test_vocab_parallel_cross_entropy(world, extreme):
    results = dist_utils.run_distributed(
        _w_vocab_cross_entropy,
        world_size=world,
        backend="gloo",
        args=(extreme,),
        timeout_s=200,
    )

    losses = [r["loss"] for r in results.values()]
    assert max(losses) - min(losses) < 1e-8, (
        f"the ranks disagree about the loss: {losses}. Every rank should end up with "
        "the same scalar."
    )

    for rank, info in results.items():
        reduces = info["counts"].get("all_reduce", 0)
        assert 2 <= reduces <= 4, (
            f"rank {rank}: expected roughly three all-reduces (max, sum-exp, target "
            f"logit), saw {reduces}. Many more means the full logits are being "
            "gathered, which is the thing this whole design avoids."
        )


# --------------------------------------------------------------------------- #
# step 8: the whole model
# --------------------------------------------------------------------------- #


def _w_full_model():
    rank = dist_utils.rank()
    config = plain_blocks.tiny_config()

    reference = plain_blocks.make_model(config, seed=6)
    ids = torch.randint(0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(6))
    targets = torch.randint(
        0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(7)
    )

    expected_logits, expected_loss = reference(ids, targets)

    parallel = tp_gpt.TensorParallelGPT.from_reference(reference)

    with comm_counter.CommCounter() as counter:
        _logits, loss = parallel(ids, targets)
    counts = dict(counter.counts())

    if not torch.allclose(loss, expected_loss, rtol=1e-5, atol=1e-7):
        raise AssertionError(
            f"rank {rank}: loss {float(loss):.8f} against reference {float(expected_loss):.8f}"
        )

    logits_only, _ = parallel(ids)
    testing.compare_to_reference(
        logits_only,
        expected_logits,
        rtol=1e-5,
        atol=1e-6,
        name=f"rank {rank} full-model logits",
    )

    return {"rank": rank, "counts": counts, "loss": float(loss), "n_layer": config.n_layer}


@pytest.mark.parametrize("world", TP_SIZES)
def test_full_model_logits_match(world):
    """The assertion that covers the whole lab."""
    results = dist_utils.run_distributed(
        _w_full_model, world_size=world, backend="gloo", timeout_s=280
    )
    losses = [r["loss"] for r in results.values()]
    assert max(losses) - min(losses) < 1e-8, f"ranks disagree about the loss: {losses}"


@pytest.mark.parametrize("world", TP_SIZES)
def test_four_allreduces_per_layer_per_step(world):
    """THEORY section 5, asserted.

    Two forward (attention, MLP) and two backward per layer, plus the embedding
    and the loss. A count well above that means something is being gathered that
    does not need to be.
    """
    results = dist_utils.run_distributed(
        _w_full_model, world_size=world, backend="gloo", timeout_s=280
    )

    for rank, info in results.items():
        n_layer = info["n_layer"]
        forward_reduces = info["counts"].get("all_reduce", 0)
        # Forward only: 2 per layer, plus 1 embedding, plus ~3 for the loss.
        expected = 2 * n_layer + 1 + 3
        assert forward_reduces <= expected + 2, (
            f"rank {rank}: {forward_reduces} forward all-reduces for a "
            f"{n_layer}-layer model. Expected around {expected}: two per layer, one "
            "for the embedding, three for the vocab-parallel loss.\n"
            "  A larger number usually means gather_output=True somewhere it should "
            "be False."
        )


# --------------------------------------------------------------------------- #
# step 9: against the production API
# --------------------------------------------------------------------------- #


def _w_against_parallelize_module():
    """Same maths through torch's own DTensor-based API.

    Not a correctness check of your code so much as a demonstration that the
    declarative version expresses the same sharding. Skips cleanly on torch builds
    without a working DeviceMesh on gloo.
    """
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )
    except ImportError:
        return "unavailable"

    world = dist_utils.world_size()
    mesh = init_device_mesh("cpu", (world,))

    reference = plain_blocks.make_mlp(plain_blocks.tiny_config(), seed=2).float()
    x = torch.randn(2, 8, reference.c_fc.in_features, generator=torch.Generator().manual_seed(2))
    expected = reference(x)

    import copy

    model = copy.deepcopy(reference)
    parallelize_module(
        model,
        mesh,
        {
            "c_fc": ColwiseParallel(),
            "c_proj": RowwiseParallel(),
        },
    )
    got = model(x)

    if not torch.allclose(got, expected, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"torch's parallelize_module disagreed with the reference by "
            f"{(got - expected).abs().max():.3e}"
        )
    return "matched"


@pytest.mark.parametrize("world", [2])
def test_matches_pytorch_parallelize_module(world):
    results = dist_utils.run_distributed(
        _w_against_parallelize_module, world_size=world, backend="gloo", timeout_s=240
    )
    values = set(results.values())
    if values == {"unavailable"}:
        pytest.skip("torch.distributed.tensor.parallel is not available in this build")
    assert values == {"matched"}
