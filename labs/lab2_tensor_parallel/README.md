# Lab 2 — Megatron-Style Tensor Parallelism

The hardest lab and the most valuable. You are going to split a matrix multiply across
devices, keep autograd correct while doing it, and end up with a GPT whose logits match
the single-process reference to five decimal places.

Megatron-LM §3 is the blueprint. Read it twice: once for the algebra, once for the `f`
and `g` operators.

---

## 1. Objectives

1. Implement `ColumnParallelLinear` and `RowParallelLinear` with correct autograd.
2. Apply tensor parallelism to the MLP and attention blocks of the reference GPT and
   match single-process outputs exactly.
3. Implement vocab-parallel embedding and vocab-parallel cross-entropy.
4. Explain from first principles why tensor parallelism does not scale past a fast
   interconnect domain.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | The sharding algebra by hand | 45 min |
| 1 | Mappings, layers, blocks, vocab, full model | 8–14 h |
| 2 | Communication scaling on 2xT4 | 45 min |

Expected Tier-1 run time: about 3 minutes for the whole file. Most tests are single
forward and backward passes, not training runs.

Budget two to three days. This lab is longer than the others and it does not compress.

## 3. Required reading

About 30 pages.

- **Shoeybi et al., *Megatron-LM* §3** (arXiv 1909.08053). Eight pages. Figure 3 is the
  MLP partitioning and Figure 4 is attention. The `f` and `g` operators are defined in
  half a paragraph that repays a slow read.
- **Ultra-Scale Playbook, the tensor-parallelism chapter.** Better diagrams, plus the
  scaling-ceiling argument with real numbers.

Reference while coding: `megatron/core/tensor_parallel/layers.py` and `mappings.py`. Read
them *after* your version passes, not before.

Optional: Korthikanti et al. for sequence parallelism (the stretch goal), and the PyTorch
`DTensor` / `parallelize_module` tutorial for the production API you will compare against
in step 9.

## 4. ⚠ The load-bearing insight: conjugate operators

A collective inside a model is not a plain function call. It has to participate in
autograd, which means it has to be a `torch.autograd.Function` with a hand-written
backward.

Two of them, and they are duals of each other:

| Operator | Forward | Backward |
|---|---|---|
| `f` = `CopyToTensorParallelRegion` | identity | **all-reduce** |
| `g` = `ReduceFromTensorParallelRegion` | **all-reduce** | identity |

Plus `ScatterToTensorParallelRegion` (split forward, all-gather backward) and
`GatherFromTensorParallelRegion` (all-gather forward, split backward) for the
sequence-parallel stretch goal.

**Swapping `f` and `g` produces a model that trains, shows a plausible falling loss, and
computes wrong gradients.** The forward outputs are still correct, so a forward-only test
sees nothing. This is the most instructive bug in the repo, and `tests/test_negative.py`
demonstrates that the suite catches it.

`THEORY.md` makes you derive *why* a forward all-reduce implies an identity backward
before you write either.

One more trap: an in-place `dist.all_reduce` on a tensor autograd is tracking, called
outside a custom `Function`, silently breaks the graph. The gradient that flows back is
the gradient of the *pre-reduce* tensor, which is neither an error nor the right answer.

## 5. The math

[THEORY.md](THEORY.md) has seven derivations. Do them all. Step 1 of the build ladder is
non-negotiable and it is pure algebra.

The short version:

**Column parallel.** `W = [W₁ | W₂]`, so `XW = [XW₁ | XW₂]`. The output is sharded along
the feature dimension. **The bias is sharded too.**

**Row parallel.** `W = [W₁; W₂]` stacked, with `X = [X₁ | X₂]` split, so
`XW = X₁W₁ + X₂W₂`. That is a **sum**, so it needs a forward all-reduce. **The bias is
not sharded and must be added once, after the reduction.** Add it on every rank before
the reduce and you get `N × bias`.

**Column then row in the MLP.** The column-parallel first layer emits activations already
sharded on the feature dimension. GELU is elementwise, so it applies to a shard with no
communication. The row-parallel second layer consumes the shard directly. **One forward
all-reduce for the whole block.** Derive why row-then-column needs two.

**Attention.** Shard QKV column-wise *by head*, so each rank owns whole heads and computes
attention locally. Shard the output projection row-wise. Constraint:
`n_head % tp_size == 0`. Derive what breaks without it.

**Communication count.** Per transformer layer: 2 all-reduces forward (attention, MLP)
and 2 backward. **Four per layer per step**, each on a `(b, s, h)` tensor.

**The scaling ceiling, which is the economically important part.** Each all-reduce's
logical tensor payload is proportional to `b·s·h` and is **independent of the TP degree
`t`**, while per-GPU compute falls as `1/t`. Actual ring link traffic includes the
bounded `2(t-1)/t` factor, so it is not exactly constant. The
compute-to-communication ratio still deteriorates with `t`. Combine that with the ring
all-reduce cost from Lab 0 and solve for the `t` at which communication exceeds
computation. That is why tensor parallelism
stays inside an NVLink domain, typically 8 GPUs, and data or pipeline parallelism is used
across nodes.

**Vocab-parallel cross-entropy.** Local logits are `(b, s, V/t)`. The full logit tensor is
often the largest activation in the model, so gathering it is the thing to avoid. The
numerically stable distributed softmax is five steps and `starter/vocab.py` lists them as
postconditions.

## 6. ⚠ Dropout and RNG under tensor parallelism

Activations that are **replicated** across the TP group must use the **same** RNG state,
or the ranks diverge. Activations that are **sharded** must use **different** RNG state,
or the effective dropout pattern is wrong: every rank masks the same positions of its own
shard, which is not the same distribution as masking the full tensor once.

Megatron maintains a separate model-parallel RNG tracker for exactly this. This lab sets
`dropout=0` by default and makes the tracker a stretch goal. That is a simplification and
it is documented rather than hidden, because it is a real correctness trap that costs
people real accuracy.

## 7. Build steps

Strictly in order. Do not attempt the full model before the layer tests pass.

**Step 1 → `test_00_math.py`** — pure algebra, no distribution. Split a weight matrix by
hand and assert `XW == cat([XW₁, XW₂])` and `XW == X₁W₁ + X₂W₂`. In float64. This step is
non-negotiable: the algebra has to be solid before communication enters the picture.

**Step 2 → `test_01_semantics.py::test_mappings_gradcheck`** — the four autograd
`Function`s in `starter/mappings.py`. Verified with `torch.autograd.gradcheck` in float64.

**Step 3 → `test_01_semantics.py::test_column_parallel_linear`, `test_row_parallel_linear`**
— `starter/layers.py`. Given a reference `nn.Linear`, shard its weights and assert both
the output and the input gradient match to `rtol=1e-5`.

**Step 4 → `test_01_semantics.py::test_parallel_mlp`** — `ParallelMLP`. Matches the
reference MLP, and `CommCounter` shows exactly one forward all-reduce.

**Step 5 → `test_01_semantics.py::test_parallel_attention`** — `ParallelCausalSelfAttention`.
Head-divisibility assertion with a message naming both numbers.

**Step 6 → `test_01_semantics.py::test_vocab_parallel_embedding`** — shard the vocabulary,
mask out-of-range indices to zero, all-reduce.

**Step 7 → `test_01_semantics.py::test_vocab_parallel_cross_entropy`** — matches
`F.cross_entropy` to 1e-5, including targets that fall in different ranks' shards.

**Step 8 → `test_01_semantics.py::test_full_model_logits_match`** — load the reference
weights, shard them, assert the logits match across the whole model. Then train 100 steps
and compare loss curves.

**Step 9 → `test_01_semantics.py::test_matches_pytorch_parallelize_module`** — the same
math through PyTorch's `ColwiseParallel`/`RowwiseParallel` on a `DeviceMesh`.

## 8. Verification

```bash
python tasks.py test LAB=2
```

## 9. Instrumentation & deliverable

`runs/lab2_tensor_parallel/tp_scaling.png`: logical collective payload per step against
`tp_size`, measured against the `b·s·h` prediction, with the compute-to-communication
ratio on a second axis.

Write `runs/lab2_tensor_parallel/writeup.md`:

1. The scaling-ceiling derivation with real numbers, for a pair of T4s over PCIe against
   an 8-GPU NVLink node. At what `t` does each become communication-bound?
2. Measured logical collective payload per step against your prediction. If they differ, which
   collective did you forget to count?
3. Why would the loss curve alone never have revealed an `f`/`g` swap? Be specific about
   what is still correct after the swap and what is not.
4. Vocab-parallel cross-entropy avoids materialising the full `(b, s, V)` logit tensor.
   Compute that tensor's size for the `tier2` preset at batch 8, sequence 512, and
   compare it to the model's parameter memory.
5. Your TP model is slower than the single-process reference at Tier 1. Give the number
   and account for it.

## 10. Reconciliation

```bash
python tasks.py reconcile LAB=2
```

## 11. Stretch goals

- **Sequence parallelism** (Korthikanti et al.). Replace the all-reduce with
  reduce-scatter plus all-gather around the LayerNorm regions. Identical total bytes,
  lower activation memory. Measure the reduction; the formula in `common/memory.py`
  predicts it.
- **The model-parallel RNG tracker.** Fix the dropout problem from section 6 properly,
  with separate RNG state for sharded and replicated regions.
- **Async tensor parallelism.** Overlap the all-gather with the matmul that consumes it.
- **`DTensor` from scratch.** Reimplement `ColumnParallelLinear` using `DTensor` placements
  rather than manual sharding, and compare the code size.

## 12. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Forward matches, backward does not | `f` and `g` swapped | The signature bug of this lab. See section 4. |
| Output is exactly `tp_size` times too large | Bias added on every rank before a row-parallel all-reduce | Add the bias once, after the reduce. |
| Gradients are silently wrong, no error | Bare `dist.all_reduce` on a tracked tensor | Every in-model collective goes through a `torch.autograd.Function`. |
| `gradcheck` fails on a `Function` that looks right | Running in float32 | `gradcheck` needs float64. Its default tolerances assume it. |
| Attention output wrong only at `tp_size=4` | A head split across two ranks | Shard by head, not by hidden dimension. Assert `n_head % tp_size == 0`. |
| Embedding output has extra nonzeros | Out-of-range indices not masked before lookup | Mask to zero, look up, all-reduce. |
| Cross-entropy off by a small constant | The global max was not subtracted before `exp` | All-reduce MAX first. See the five steps in `starter/vocab.py`. |
| `tiny` preset fails at `tp_size=4` but the algebra is right | `tiny` has 4 heads, so `max_tp=4` | Correct behaviour. Use `small` for `tp_size=8`. |
| Loss falls nicely and the logits do not match | You are looking at the wrong evidence | The logits are the test. The loss is not. |

## 13. Where this lives in production

- **`megatron/core/tensor_parallel/layers.py`** — `ColumnParallelLinear` and
  `RowParallelLinear`, the direct ancestors of what you wrote. Note how much of the file
  is fused kernels and gradient-accumulation fusion around the same core idea.
- **`megatron/core/tensor_parallel/mappings.py`** — the four `f`/`g` `Function`s, named
  `_CopyToModelParallelRegion` and friends. Your version should be recognisably the same.
- **`megatron/core/tensor_parallel/cross_entropy.py`** — the vocab-parallel loss, with the
  same five steps.
- **`megatron/core/tensor_parallel/random.py`** — the RNG tracker from section 6.
- **`torch.distributed.tensor.parallel`** — `ColwiseParallel`, `RowwiseParallel`, and
  `parallelize_module`. The modern API expresses the same sharding declaratively over a
  `DeviceMesh`, and step 9 compares against it.
- **`picotron/tensor_parallel/`** — the readable version, roughly 250 lines.
