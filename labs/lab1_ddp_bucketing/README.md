# Lab 1 — DDP from Scratch: Gradient Bucketing & Overlap

You are going to rebuild `torch.nn.parallel.DistributedDataParallel`. Not a toy version:
the real semantics, including bucketing in reverse registration order, asynchronous
collectives overlapped with the backward pass, `no_sync` for gradient accumulation, and
the unused-parameter hang that commonly appears when production models add conditional
branches.

The PyTorch DDP paper (Li et al., VLDB 2020) §3 is the blueprint. This lab is that
section, implemented.

---

## 1. Objectives

1. Build a `MyDDP` wrapper matching `DistributedDataParallel` semantics closely enough
   that gradients agree elementwise.
2. Overlap gradient communication with backward computation, using per-parameter
   autograd hooks and size-triggered asynchronous collectives.
3. Derive the bucket-size trade-off and validate it against a measured sweep.
4. Handle gradient accumulation, unused parameters, and the learning-rate consequence of
   a larger effective batch.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | Bucket flatten/unflatten, assignment order, accumulation algebra | 30–45 min |
| 1 | The four DDP variants and their tests | 5–8 h |
| 2 | Overlap and scaling on an optional two-GPU host | about 45 min |

Expected Tier-1 run time: **about 2 minutes** for a 20-step training equivalence check at
world size 4, and under 3 minutes for the whole file.

**Read this before you measure anything.** Your DDP will be *slower* than single-process
at Tier 1, probably by 2 to 5 times. Nothing is wrong. Four CPU ranks on one laptop share
one memory bus and one set of cores, so you have added communication without adding
compute. Tier 1 answers "is it correct". Tier 2 answers "is it fast". Do not conflate
them, and do not go looking for a bug that is not there.

## 3. Required reading

About 25 pages.

- **Li et al., *PyTorch Distributed*, §3** (arXiv 2006.15704). Twelve pages, and it is
  the entire lab. §3.2.2 on gradient bucketing and §3.2.3 on overlap are the parts you
  will reread. Pay attention to the argument for reverse order and to the bucket-rebuild
  step after the first iteration.
- **Ultra-Scale Playbook, the data-parallelism chapter.** The same material with better
  diagrams, plus the ZeRO framing that Lab 1B picks up.
- **Goyal et al. §2** (arXiv 1706.02677). Three pages: the linear scaling rule and
  gradual warmup. Short, and it is why your loss curve does not match if you forget that
  N-way DDP multiplies the effective batch by N.

Reference while coding: `torch/nn/parallel/distributed.py`, and the `register_hook`
family in the autograd docs. Optional: the PyTorch DDP design note, and
`torch/csrc/distributed/c10d/reducer.cpp` if you want to see how much of this is really
C++.

## 4. The math

[THEORY.md](THEORY.md) has five derivations with blanks. Do them before step 3, and fill
[predictions.yaml](predictions.yaml) before running `bench/bucket_sweep.py`.

The short version:

**Overlap.** Serial execution costs `T_bwd + T_comm`. Perfect overlap costs
`max(T_bwd, T_comm)` plus a tail, because the last bucket cannot start until the last
gradient exists. Write the expression for the exposed time and you have the whole
performance model of DDP.

**Bucket size.** With `P` bytes of gradients in `K` buckets, you pay `K` fixed latencies
and the tail shrinks as buckets get smaller. Two opposing forces, so there is an optimum.
Derive its direction and predict the best size from the α and β you fitted in Lab 0.
PyTorch's default is 25 MiB. Work out why that is sensible on NVLink and why it might not
be on gloo.

**Reverse order.** Gradients become ready in reverse order of the forward pass. Bucketing
in reverse `model.parameters()` order approximates readiness order, so bucket 0 fills
first and its collective launches earliest. Work out what goes wrong with forward order.
(Real DDP goes further and *rebuilds* the buckets after iteration one using the observed
readiness order. Note why that is more robust than assuming reverse registration order.)

**Accumulation.** Show that averaging gradients over `k` micro-batches with the loss
scaled by `1/k` equals one batch of size `kb`, for mean-reduced losses. Then identify
what breaks the equivalence.

## 5. ⚠ The API correction

This is the single most important API detail in the lab. Older examples may still use
the deprecated hook below, so check the current autograd API rather than copying one
unchanged.

`nn.Module.register_backward_hook` is **deprecated and does not do what you need**. It is
a module-level hook with documented incorrect behaviour for several module types, and
crucially it does not tell you when an individual *parameter's* gradient is ready. Do not
use it.

Three things that do work:

- `param.register_post_accumulate_grad_hook(fn)` fires after `param.grad` has been
  accumulated. Clean and correct. Needs torch 2.1 or later.
- `param.register_hook(fn)` fires when the gradient *with respect to that tensor* is
  computed. Earlier than accumulation, and it can fire more than once for a shared
  parameter. Usable, subtler.
- The autograd-graph approach real DDP uses: grab the `AccumulateGrad` node via
  `param.expand_as(param).grad_fn.next_functions[0][0]` and register on that. This is
  what `reducer.cpp` does. Implementing it once makes the PyTorch source readable.

Pick one. The tests check behaviour, not which API you chose, and the trade-offs above
are the whole of what you need to decide.

## 6. Build steps

**Step 1 → `test_00_math.py::test_bucket_roundtrip`**
`starter/bucket.py::GradientBucket`. Flatten a group of gradients into one contiguous
buffer, and unflatten back into `.grad` views.

⚠ Use `torch._utils._flatten_dense_tensors` and `_unflatten_dense_tensors`, or an
explicit contiguous buffer with per-parameter views. Do **not** `torch.cat` and copy back:
that doubles peak memory during the flush and silently breaks any view someone else holds
on `.grad`.

**Step 2 → `test_00_math.py::test_bucket_assignment_*`**
`assign_buckets(params, bucket_bytes)`. Reverse registration order, size-triggered.

**Step 3 → `test_01_semantics.py::test_simple_ddp_matches_reference`**
`starter/simple_ddp.py::SimpleDDP`. No bucketing, no overlap. Broadcast parameters from
rank 0 at construction. After `loss.backward()`, loop over parameters, all-reduce each
gradient, divide by world size. Twenty training steps must match the single-process loss
curve.

Get the division right. Summing without dividing gives you an effective learning rate N
times too large, the loss still falls, and you find out three labs later.

**Step 4 → `test_01_semantics.py::test_hooked_ddp_*`**
`starter/hooked_ddp.py::HookedDDP`. Per-parameter hooks, still synchronous. Same loss
curve, and `CommCounter` should now show one all-reduce per parameter.

**Step 5 → `test_01_semantics.py::test_bucketed_ddp_collective_count`**
`starter/bucketed_ddp.py::BucketedDDP`. Group gradients into buckets, flush a bucket when
it fills. The count should drop from one per parameter to `ceil(total_bytes /
bucket_bytes)`.

**Step 6 → `test_01_semantics.py::test_async_overlap`**
Make the flush asynchronous: `async_op=True`, keep the handle, carry on with the backward
pass, and `wait()` on every handle in a `finalize()` called before `optimizer.step()`.

⚠ Reading `.grad` before `wait()` gives you a *partially reduced* tensor. Right shape,
right dtype, plausible values, slightly wrong training. `test_negative.py` demonstrates
this on purpose.

**Step 7 → `test_01_semantics.py::test_no_sync_*`**
`no_sync()` context manager. Inside it, hooks accumulate gradients without communicating.
`k` micro-steps then one sync step should equal one large-batch step, with `1/k` the
collectives.

**Step 8 → `test_01_semantics.py::test_unused_parameters_do_not_hang`**
A model with a conditionally-unused branch. Parameters that never receive a gradient
never fire their hook, so their bucket never fills, so `finalize()` waits forever and the
job hangs with no error. This is the classic production DDP hang. Detect and handle it.

**Step 9 → the deliverable**
`python tasks.py bench LAB=1` sweeps bucket sizes and renders the overlap Gantt chart.

## 7. Verification

```bash
python tasks.py test LAB=1
```

Green looks like:

```
tests/test_00_math.py .............        [ 30%]
tests/test_01_semantics.py .............   [ 88%]
tests/test_negative.py ...                 [ 95%]
tests/test_02_perf.py ssss                 [100%]
======== 38 passed, 4 skipped in 148s ========
```

## 8. Instrumentation & deliverable

`python tasks.py bench LAB=1` writes to `runs/lab1_ddp_bucketing/`:

- `bucket_sweep.png` — step time against bucket size {0.5, 1, 5, 10, 25, 100} MiB, with
  your predicted optimum marked.
- `overlap_gantt.png` — compute and communication intervals per rank. The question the
  chart answers: are the orange bars underneath the blue ones, or after them?
- `measurements.yaml` for reconciliation.

Write `runs/lab1_ddp_bucketing/writeup.md`:

1. Where did you predict the optimal bucket size, where did you measure it, and what
   does the difference tell you about which term in the model is wrong?
2. From the Gantt chart, what fraction of your communication was exposed? What is the
   theoretical floor for that fraction, and why is it not zero?
3. Your DDP is slower than single-process at Tier 1. Give the numbers and explain
   precisely where the time goes. Do not write "communication overhead"; say which
   communication, how many bytes, and how that compares to the compute it was supposed
   to hide behind.
4. Bucketing in forward registration order instead of reverse: what happens to the
   overlap, and why? Reason it out from the readiness order, then try it and see.
5. Real DDP rebuilds its buckets after the first iteration rather than trusting reverse
   registration order. Name a model architecture where reverse order is a bad
   approximation of readiness order.

## 9. Reconciliation

```bash
python tasks.py reconcile LAB=1
```

## 10. Stretch goals

- **The `AccumulateGrad` approach.** Register on the autograd node the way
  `reducer.cpp` does. Afterwards, read `reducer.cpp`. It becomes legible.
- **Bucket rebuild.** Record the actual gradient-readiness order during iteration one,
  then reassign buckets to match. Measure whether it helps on this model. It probably
  will not, and understanding why is the interesting part.
- **Gradient compression.** All-reduce in fp16 and accumulate the error locally. Measure
  the loss-curve deviation against the halved bytes.
- **Comparison against `DistributedDataParallel(bucket_cap_mb=...)`.** Same sweep, real
  DDP, on the same model. Where does the shape of the curve differ, and why?

## 11. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Loss curves match at step 0, diverge after | Gradients summed but not divided by world size | Divide. The effective LR was N times too high. |
| Loss curves differ at step 0 | Parameters not broadcast from rank 0 | `dist_utils.sync_module_params` in the constructor. |
| Loss falls but slightly slower than reference | `.grad` read before the async handle was waited on | `finalize()` before `optimizer.step()`. See `test_negative.py`. |
| Job hangs at the end of a step, no error | A parameter never got a gradient, so its bucket never filled | Step 8. This is the classic one. |
| Hooks fire twice for one parameter | Weight tying, or `register_hook` instead of the post-accumulate variant | Understand which; both are legitimate and they need different handling. |
| Collective count is right, bytes are double | Flushing the bucket and then also all-reducing individual gradients | Check you removed the step-3 path. |
| `CommCounter` shows zero collectives | Calling `torch.distributed` directly instead of `common.collectives` | Use the wrappers; that is what makes the counting possible. |
| Everything correct, Tier 1 slower than 1 process | Expected. See section 2. | Not a bug. |

## 12. Where this lives in production

- **`torch/nn/parallel/distributed.py`** — the python surface you are reimplementing.
  Look at `_ddp_init_helper` for bucket construction and `_sync_params`.
- **`torch/csrc/distributed/c10d/reducer.cpp`** — where the real work is.
  `Reducer::mark_variable_ready` is your hook callback. `Reducer::finalize_backward` is
  your `finalize`. `Reducer::rebuild_buckets` is the stretch goal. Reading this file
  after finishing the lab is thirty of the highest-value minutes in the curriculum.
- **`picotron/data_parallel/data_parallel.py`** — the same thing in 200 readable lines.
  Diff it against yours.
- **DeepSpeed's `DeepSpeedEngine`** for how the same machinery gets extended to ZeRO,
  which is Lab 1B.
