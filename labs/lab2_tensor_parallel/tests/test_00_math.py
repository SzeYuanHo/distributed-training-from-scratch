"""Tier 0: the sharding algebra, by hand, in float64, in one process.

Build step 1, and it is non-negotiable. Before any communication exists to confuse
the picture, you should be able to split a weight matrix and show the pieces
recombine into the original answer.

These tests do not import your starter code at all. They are here so that when the
distributed tests fail you can be certain the algebra was not the problem.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from labs.lab2_tensor_parallel.reference import plain_blocks

pytestmark = pytest.mark.tier0

TOL = dict(rtol=1e-12, atol=1e-12)  # float64, so we can be strict


# --------------------------------------------------------------------------- #
# column parallel: concatenation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [2, 4, 8])
def test_column_split_concatenates(n_shards):
    """XW == cat([XW_1, ..., XW_t]) along the output dimension."""
    torch.manual_seed(0)
    x = torch.randn(6, 32, dtype=torch.float64)
    weight = torch.randn(64, 32, dtype=torch.float64)  # torch layout: (out, in)

    full = F.linear(x, weight)

    shards = plain_blocks.split_weight_columns(weight, n_shards)
    pieces = [F.linear(x, shard) for shard in shards]
    recombined = torch.cat(pieces, dim=-1)

    torch.testing.assert_close(recombined, full, **TOL)
    assert pieces[0].shape[-1] == 64 // n_shards


@pytest.mark.parametrize("n_shards", [2, 4])
def test_column_split_shards_the_bias_too(n_shards):
    """Column-parallel bias is sharded alongside the output it belongs to."""
    torch.manual_seed(1)
    x = torch.randn(6, 32, dtype=torch.float64)
    weight = torch.randn(64, 32, dtype=torch.float64)
    bias = torch.randn(64, dtype=torch.float64)

    full = F.linear(x, weight, bias)

    w_shards = plain_blocks.split_weight_columns(weight, n_shards)
    b_shards = list(bias.chunk(n_shards, dim=0))
    recombined = torch.cat([F.linear(x, w, b) for w, b in zip(w_shards, b_shards)], dim=-1)

    torch.testing.assert_close(recombined, full, **TOL)


# --------------------------------------------------------------------------- #
# row parallel: summation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [2, 4, 8])
def test_row_split_sums(n_shards):
    """XW == X_1 W_1 + X_2 W_2 + ...

    A sum, not a concatenation. That single fact is why row-parallel needs a
    forward all-reduce and column-parallel does not.
    """
    torch.manual_seed(2)
    x = torch.randn(6, 64, dtype=torch.float64)
    weight = torch.randn(32, 64, dtype=torch.float64)  # (out=32, in=64)

    full = F.linear(x, weight)

    w_shards = plain_blocks.split_weight_rows(weight, n_shards)
    x_shards = list(x.chunk(n_shards, dim=-1))
    recombined = sum(F.linear(xs, ws) for xs, ws in zip(x_shards, w_shards))

    torch.testing.assert_close(recombined, full, **TOL)


@pytest.mark.parametrize("n_shards", [2, 4])
def test_row_split_bias_added_once(n_shards):
    """The bias goes on after the sum, not before it.

    This test is worth reading rather than just running, because the wrong version
    produces a tensor of the right shape with entirely plausible values.
    """
    torch.manual_seed(3)
    x = torch.randn(6, 64, dtype=torch.float64)
    weight = torch.randn(32, 64, dtype=torch.float64)
    bias = torch.randn(32, dtype=torch.float64)

    full = F.linear(x, weight, bias)

    w_shards = plain_blocks.split_weight_rows(weight, n_shards)
    x_shards = list(x.chunk(n_shards, dim=-1))

    correct = sum(F.linear(xs, ws) for xs, ws in zip(x_shards, w_shards)) + bias
    torch.testing.assert_close(correct, full, **TOL)

    # And the wrong version, so you can see what it looks like.
    wrong = sum(F.linear(xs, ws, bias) for xs, ws in zip(x_shards, w_shards))
    difference = wrong - full
    expected_excess = bias * (n_shards - 1)

    torch.testing.assert_close(
        difference,
        expected_excess.expand_as(difference),
        **TOL,
        msg="adding the bias on every rank should overshoot by exactly (t-1) biases",
    )


# --------------------------------------------------------------------------- #
# the composition that makes the MLP cheap
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [2, 4])
def test_column_then_row_needs_no_intermediate_communication(n_shards):
    """The whole argument for the column-then-row ordering, as arithmetic.

    The column layer emits a shard. GELU is elementwise, so it applies to the shard
    unchanged. The row layer consumes the shard. Nothing has to be exchanged in
    between, and the final sum is the only place ranks talk.
    """
    torch.manual_seed(4)
    x = torch.randn(6, 32, dtype=torch.float64)
    w1 = torch.randn(128, 32, dtype=torch.float64)  # (4h, h)
    w2 = torch.randn(32, 128, dtype=torch.float64)  # (h, 4h)

    full = F.linear(F.gelu(F.linear(x, w1)), w2)

    w1_shards = plain_blocks.split_weight_columns(w1, n_shards)
    w2_shards = plain_blocks.split_weight_rows(w2, n_shards)

    # Each "rank" works entirely alone until the final sum.
    partials = [F.linear(F.gelu(F.linear(x, a)), b) for a, b in zip(w1_shards, w2_shards)]

    torch.testing.assert_close(sum(partials), full, **TOL)


def test_gelu_on_a_shard_equals_the_shard_of_gelu():
    """Elementwise means elementwise. The one-line reason the above works."""
    torch.manual_seed(5)
    x = torch.randn(6, 128, dtype=torch.float64)

    full = F.gelu(x)
    sharded = torch.cat([F.gelu(chunk) for chunk in x.chunk(4, dim=-1)], dim=-1)

    torch.testing.assert_close(sharded, full, **TOL)


def test_layernorm_on_a_shard_does_not_equal_the_shard_of_layernorm():
    """The contrast, and the reason sequence parallelism exists.

    LayerNorm normalises across the feature dimension, so it needs every feature.
    Apply it to a shard and you get the normalisation of that shard, which is a
    different tensor. That is why the LayerNorms in ``ParallelBlock`` stay
    replicated rather than sharded on the hidden dimension.
    """
    torch.manual_seed(6)
    x = torch.randn(6, 128, dtype=torch.float64)

    full = F.layer_norm(x, (128,))
    naive = torch.cat([F.layer_norm(c, (32,)) for c in x.chunk(4, dim=-1)], dim=-1)

    assert not torch.allclose(naive, full, rtol=1e-6), (
        "LayerNorm applied per shard came out equal to the full LayerNorm, which "
        "would mean this test's premise is wrong"
    )


# --------------------------------------------------------------------------- #
# head sharding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [2, 4])
def test_attention_by_whole_heads_recombines(n_shards):
    """Sharding by head lets each rank compute its heads with no communication."""
    torch.manual_seed(7)
    b, t, n_head, head_dim = 2, 8, 4, 16

    q = torch.randn(b, n_head, t, head_dim, dtype=torch.float64)
    k = torch.randn(b, n_head, t, head_dim, dtype=torch.float64)
    v = torch.randn(b, n_head, t, head_dim, dtype=torch.float64)

    def attend(qq, kk, vv):
        scores = (qq @ kk.transpose(-2, -1)) / head_dim**0.5
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=-1) @ vv

    full = attend(q, k, v)

    per_rank = n_head // n_shards
    pieces = [
        attend(
            q[:, i * per_rank : (i + 1) * per_rank],
            k[:, i * per_rank : (i + 1) * per_rank],
            v[:, i * per_rank : (i + 1) * per_rank],
        )
        for i in range(n_shards)
    ]

    torch.testing.assert_close(torch.cat(pieces, dim=1), full, **TOL)


def test_qkv_fused_weight_slicing_is_not_a_plain_chunk():
    """The trap in ParallelCausalSelfAttention.load_from_reference.

    ``c_attn`` produces Q, K and V concatenated: rows 0..h-1 are Q, h..2h-1 are K,
    2h..3h-1 are V. Rank i needs its own heads' rows from all three, so a plain
    ``chunk(t, dim=0)`` hands rank 0 the whole of Q and rank 1 the whole of K and V.

    This test does not check your code. It demonstrates the failure so you can see
    what the right slicing has to achieve.
    """
    torch.manual_seed(8)
    n_head, head_dim, t = 4, 8, 2
    hidden = n_head * head_dim
    weight = torch.arange(3 * hidden * hidden, dtype=torch.float64).reshape(3 * hidden, hidden)

    naive = list(weight.chunk(t, dim=0))
    assert naive[0].shape[0] == 3 * hidden // t

    # The correct slicing: view as (3, n_head, head_dim, hidden), take this rank's
    # heads, flatten back.
    viewed = weight.view(3, n_head, head_dim, hidden)
    per_rank = n_head // t
    correct = [viewed[:, i * per_rank : (i + 1) * per_rank].reshape(-1, hidden) for i in range(t)]

    assert correct[0].shape == naive[0].shape, "same shape, which is why this is a trap"
    assert not torch.equal(correct[0], naive[0]), (
        "the naive chunk and the correct head slicing came out identical, which "
        "would mean this test's premise is wrong"
    )

    # And the correct pieces really do reassemble the original.
    reassembled = torch.cat(
        [correct[i].view(3, per_rank, head_dim, hidden) for i in range(t)], dim=1
    ).reshape(3 * hidden, hidden)
    torch.testing.assert_close(reassembled, weight, **TOL)


# --------------------------------------------------------------------------- #
# vocab sharding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_shards", [2, 4])
def test_vocab_split_embedding_sums(n_shards):
    """Mask, look up, zero the rest, sum. The pattern before any collectives exist."""
    torch.manual_seed(9)
    vocab, dim = 32, 16
    table = torch.randn(vocab, dim, dtype=torch.float64)
    ids = torch.randint(0, vocab, (2, 6))

    full = F.embedding(ids, table)

    per_rank = vocab // n_shards
    total = torch.zeros_like(full)
    for i in range(n_shards):
        start, stop = i * per_rank, (i + 1) * per_rank
        in_range = (ids >= start) & (ids < stop)
        local_ids = (ids - start).clamp(0, per_rank - 1)
        piece = F.embedding(local_ids, table[start:stop])
        piece = piece * in_range.unsqueeze(-1)  # zero the out-of-range rows
        total = total + piece

    torch.testing.assert_close(total, full, **TOL)


@pytest.mark.parametrize("n_shards", [2, 4])
def test_distributed_softmax_is_stable(n_shards):
    """The five steps of vocab-parallel cross-entropy, without any distribution.

    Logits deliberately include a value of 100, which overflows ``exp`` in fp32 and
    is exactly why step 1 subtracts the global maximum.
    """
    torch.manual_seed(10)
    batch, vocab = 4, 32
    logits = torch.randn(batch, vocab, dtype=torch.float64) * 5
    logits[0, 7] = 100.0
    targets = torch.randint(0, vocab, (batch,))

    expected = F.cross_entropy(logits, targets)

    per_rank = vocab // n_shards
    shards = [logits[:, i * per_rank : (i + 1) * per_rank] for i in range(n_shards)]

    # step 1: global max
    global_max = torch.stack([s.max(dim=-1).values for s in shards]).max(dim=0).values
    # step 2: global sum of exp
    global_sum = sum((s - global_max.unsqueeze(-1)).exp().sum(dim=-1) for s in shards)
    # step 3: the target logit, contributed by whichever rank owns it
    target_logit = torch.zeros(batch, dtype=torch.float64)
    for i, shard in enumerate(shards):
        start, stop = i * per_rank, (i + 1) * per_rank
        owns = (targets >= start) & (targets < stop)
        local = (targets - start).clamp(0, per_rank - 1)
        picked = shard.gather(1, local.unsqueeze(1)).squeeze(1)
        target_logit = target_logit + picked * owns
    # step 4: the loss
    loss = (global_sum.log() + global_max - target_logit).mean()

    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-10)


def test_collective_payload_is_independent_of_tp_degree():
    """The scaling-ceiling argument, as arithmetic.

    The logical tensor payload per all-reduce is proportional to b*s*h and does
    not depend on t. Ring link traffic adds 2(t-1)/t; the test isolates the
    payload invariant that CommCounter records. Per-rank compute falls as 1/t,
    so the compute-to-payload ratio degrades linearly.

    That is why tensor parallelism lives inside an NVLink domain.
    """
    b, s, h, bytes_per_elem = 8, 256, 384, 4
    payload = b * s * h * bytes_per_elem

    for t in (2, 4, 8, 16):
        # The all-reduced tensor is always (b, s, h), whatever t is.
        assert payload == b * s * h * bytes_per_elem

        compute_per_rank = 24 * b * s * h * h / t  # rough FLOPs for one layer
        ratio = compute_per_rank / payload

        expected = (24 * b * s * h * h / 2) / payload * (2 / t)
        assert ratio == pytest.approx(expected, rel=1e-9), (
            f"at t={t} the compute-to-communication ratio should scale as 1/t"
        )
