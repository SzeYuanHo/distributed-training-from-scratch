# Lab 0 — Collectives & Cost Models

Everything else in this repo is built from these six operations. Do this lab first and
Labs 1 through 3 never have to hand-wave "and then we all-reduce".

---

## 1. Objectives

By the end you can, without looking anything up:

1. Implement `broadcast`, `reduce`, `all_reduce`, `reduce_scatter`, `all_gather`, and
   `all_to_all` using only point-to-point `send` and `recv`.
2. Implement all-reduce three ways (naive gather-to-root, recursive halving-doubling,
   ring) and say which wins where.
3. Fit the α–β cost model to your own hardware and use it to *predict* a collective's
   latency before you run it.
4. Explain why a ring all-reduce is bandwidth-optimal, and why NCCL still switches to a
   tree for small messages.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | `ring_allreduce_numpy` in one process | 20–30 min |
| 1 | The six primitives on gloo, plus tests | 3–5 h |
| 2 | NCCL sweep on an optional two-GPU host | about 30 min |

Expected Tier-1 run time for the whole test file: **under 90 seconds** on a laptop at
world size 4. If it takes minutes, check `OMP_NUM_THREADS`; `python tasks.py test` sets
it for you but a bare `pytest` does not.

The `bench/sweep.py` sweep takes 2–4 minutes because it runs 3 algorithms across 17
message sizes and 3 world sizes.

## 3. Required reading

Roughly 30 pages. Read section 4 of this README first, attempt the derivations, and go
to the text when you get stuck. Reading before implementing wastes time; reading after
failing is when it sticks.

- **Grama et al., *Introduction to Parallel Computing* 2nd ed.**
  - §2.5, the communication cost model. Four pages. This is where `t_s + t_w·m` comes
    from, and it is the same α–β model everyone else uses with different letters.
  - §4.1–4.3, one-to-all broadcast, all-to-all, and their cost derivations on ring and
    hypercube topologies.
  - §4.7, all-reduce and prefix sums.
  - Free alternative: Eijkhout, *The Art of HPC vol. 2*, the collectives chapter.
- **Scaling Book, Chapter 1 (Rooflines).** Free at
  https://jax-ml.github.io/scaling-book/. Ten pages. Gives you the arithmetic-intensity
  frame that makes "is this communication-bound?" a computation instead of a guess.

Optional, clearly skippable:

- Sergeev & Del Balso, *Horovod* (arXiv 1802.05799), the ring all-reduce description.
  Short, and it is the paper that made ring all-reduce standard in deep learning.
- NCCL developer docs on algorithm selection, if you want to see how a production
  library decides between ring and tree.

Reference while coding: the `torch.distributed` docs' collective-semantics table. Keep
it open. The difference between `all_gather` and `all_gather_into_tensor` will cost you
twenty minutes otherwise.

## 4. The math

Work through [THEORY.md](THEORY.md) before writing code, and fill in
[predictions.yaml](predictions.yaml) before running `bench/sweep.py`. Both have blanks.

The short version, so you know what you are deriving:

Sending `m` bytes costs `T = α + β·m`, where α is per-message latency and β is inverse
bandwidth. For N ranks reducing a buffer of M bytes:

| Algorithm | Latency term | Bytes per rank | Bottleneck |
|---|---|---|---|
| Naive gather-to-root, then broadcast | O(N)·α | root receives (N-1)M | the root's link |
| Recursive halving-doubling | 2·log₂N·α | 2·(N-1)/N·M | latency-optimal |
| Ring (reduce-scatter + all-gather) | 2(N-1)·α | 2·(N-1)/N·M | bandwidth-optimal |

Three results to be able to state in your own words:

- Ring all-reduce moves `2·(N-1)/N·M` bytes per rank, which approaches `2M` as N grows
  and never exceeds it. **The volume is bounded in N, not independent of N.** That is
  why its bandwidth term stays controlled as data-parallel world size grows.
- Ring pays `2(N-1)` latency hops against recursive doubling's `2·log₂N`. Set the two
  total costs equal and solve for the point where these simplified models cross. A real
  library may use more algorithms and topology-specific selection rules. Then measure
  yours.
- *Algorithm bandwidth* is `M/T`. For a flat-ring all-reduce, *bus bandwidth* is
  `2(N-1)/N · M/T`. NVIDIA's `nccl-tests` reports both `algbw` and the derived `busbw`.
  The latter is not necessarily literal wire rate for hierarchical or hardware-assisted
  algorithms.

## 5. Build steps

Each step names the test it makes pass. Work in order; each one builds on the last.

**Step 1 → `test_00_math.py::test_ring_matches_sum`**
Implement `starter/ring_numpy.py::ring_allreduce_numpy`. N ranks simulated as a list of
numpy arrays in one process. No launcher, no ports, no hangs. Prove the algorithm
before any infrastructure exists to hide bugs in.

Handle buffer lengths that do not divide evenly by N. The test includes length 7 on 3
ranks specifically to catch a chunking scheme that only works when it divides.

**Step 2 → `test_01_semantics.py::test_world_size_and_ranks`**
Nothing to implement. Confirms `run_distributed` spawns ranks correctly and each one
sees a distinct rank. If this fails, the problem is your environment, not your code;
run `python tasks.py doctor`.

**Step 3 → `test_01_semantics.py::test_broadcast_*`**
Implement `broadcast_linear` and `broadcast_tree` in `starter/p2p_collectives.py`.
Linear first: rank 0 sends to everyone in a loop. Then the tree version, and compare the
message counts the `CommCounter` reports against your prediction.

**Step 4 → `test_01_semantics.py::test_reduce_scatter_ring`, `test_all_gather_ring`**
The two halves of a ring all-reduce. Get these right independently and step 5 is four
lines.

**Step 5 → `test_01_semantics.py::test_all_reduce_ring`**
Compose them. Then check the collective count: exactly `2(N-1)` sends per rank, no more.
A correct implementation that sends `3(N-1)` times has a redundant pass in it.

**Step 6 → `test_01_semantics.py::test_all_reduce_recursive_doubling`**
Requires N to be a power of two. Assert that, with an error message naming the world
size, because the failure mode otherwise is a silent wrong answer on N=3.

**Step 7 → `test_01_semantics.py::test_all_to_all`**
The distributed transpose. Rank i's j-th chunk ends up at rank j's i-th slot.

**Step 8 → `test_00_math.py::test_cost_model_*`**
Implement `starter/cost_model.py`: least-squares fit of α and β, prediction for each
algorithm, and the crossover point.

**Step 9 → the deliverable**
Run `python tasks.py bench LAB=0`. It sweeps 2^10 to 2^26 bytes across world sizes
{2,4,8} and all three algorithms, fits separate achieved-bandwidth terms, and plots
measured against predicted with any finite fitted ring/recursive-doubling crossover
marked.

## 6. Verification

```bash
python tasks.py test LAB=0
```

Green looks like this:

```
labs/lab0_collectives/tests/test_00_math.py ..........        [ 45%]
labs/lab0_collectives/tests/test_01_semantics.py ..........   [ 95%]
labs/lab0_collectives/tests/test_02_perf.py sss               [100%]
======== 20 passed, 3 skipped in 47.2s ========
```

The three skips are Tier 2, and they skip automatically because you do not have two
CUDA devices. That is correct, not a failure.

One test is expected to `xfail`: `test_symmetric_send_deadlocks`. It demonstrates that
naive symmetric send-then-receive hangs, wrapped in a timeout. If it ever *passes*, your
transport is buffering the whole message and the demonstration is not showing what it
claims. Read the test's docstring.

## 7. Instrumentation & deliverable

`python tasks.py bench LAB=0` writes to `runs/lab0_collectives/`:

- `cost_model.png`, measured latency against your fitted α–β prediction, all three
  algorithms, log-log, with each finite fitted ring/recursive-doubling crossover marked.
- `measurements.yaml`, the raw numbers, which `tasks.py reconcile` reads.

Write `runs/lab0_collectives/writeup.md` answering these in your own words. Short answers, a
paragraph each:

1. What are your fitted α and β for gloo on localhost? What physical thing does each
   correspond to, and does the magnitude make sense?
2. Where did you predict the ring/tree crossover, and where did you measure it? If they
   differ by more than 2x, what did the model leave out?
3. Ring all-reduce moves `2(N-1)/N·M` bytes per rank. Explain to someone who has not
   read this why that quantity stays below `2M` as N grows, and why that matters for
   data parallelism at 512 GPUs.
4. Your `broadcast_tree` sends fewer messages than `broadcast_linear` but the same total
   bytes. Under what conditions is the tree slower anyway?
5. `nccl-tests` reports both algorithm bandwidth and derived bus bandwidth. If a
   colleague tells you their all-reduce hit "180 GB/s on 8 GPUs", what do you need to
   ask before you can compare it to anything?

## 8. Reconciliation

```bash
python tasks.py reconcile LAB=0
```

Prints your `predictions.yaml` against what `bench/sweep.py` measured, with percent
error and a band. Anything over 50% off is worth a paragraph in the writeup. The
prediction being wrong is not the failure; not knowing why is.

## 9. Stretch goals

Explicitly optional. Skip all of these and the lab is still complete.

- **Segmented ring.** Split large buffers into segments and pipeline them, so the
  reduce-scatter of segment k overlaps the all-gather of segment k-1. This is what NCCL
  actually does.
- **Bandwidth-optimal broadcast.** Implement broadcast as scatter + all-gather and show
  it beats the tree above some size. Same trick as the ring all-reduce, applied to a
  different collective.
- **Hierarchical all-reduce.** Two-level: reduce within a node, all-reduce across nodes,
  broadcast within a node. You cannot measure the benefit on one machine, but you can
  count the messages and argue for it.
- **Tier 2 on two GPUs.** Run the same sweep with NCCL and compare the fitted α
  and β to gloo's. The difference is roughly two orders of magnitude, and seeing it is
  worth the thirty minutes.

## 10. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Test hangs, then `DistributedTimeout` | Ranks calling collectives in different orders, or a send with no matching recv | Print the op sequence per rank. Every rank must issue the same collectives in the same order. |
| Ring gives the right answer at N=2 and N=4, wrong at N=3 | Chunking assumed the buffer divides evenly | Distribute the remainder across the first `M % N` chunks. |
| Recursive doubling silently wrong at N=6 | It requires a power of two and you did not assert it | Assert in the function, with the world size in the message. |
| `Address already in use` | Stale port from a crashed run | `python tasks.py kill`. `free_port()` is used by default so this should be rare. |
| Timings are noisy and non-monotonic | No warmup, or the first call includes lazy connection setup | Discard the first 3 iterations. `bench/sweep.py` already does. |
| Everything correct but 10x slower than expected | Each rank grabbed every core | `python tasks.py test LAB=0` sets `OMP_NUM_THREADS`; a bare `pytest` does not. |
| `test_symmetric_send_deadlocks` unexpectedly passes | The message was small enough to fit in the transport buffer | Increase the tensor size in that test until it blocks. Note the threshold you found; it is the transport's buffer size. |

## 11. Where this lives in production

- `torch/csrc/distributed/c10d/ProcessGroupGloo.cpp` and `ProcessGroupNCCL.cpp`. The
  actual implementations of what you just wrote. `ProcessGroupGloo::allreduce` is
  readable.
- NCCL's `src/collectives/` and tuning logic. The simplified ring-versus-tree comparison
  in this lab is one slice of a runtime selector that can consider several algorithms,
  protocols, topologies, builds, and configuration settings.
- `NCCL_ALGO=Ring|Tree` forces two choices for a controlled comparison. Use
  `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` to inspect normal algorithm and protocol
  selection on a Tier-2 session rather than assuming a universal crossover.
- `torch.distributed.all_reduce` dispatches to `c10d::allreduce_`, which is the op the
  `CommCounter` dispatch mode intercepts. Reading `_c10d_functional` shows you how
  collectives became traceable ops.
