# Lab 3 — Pipeline Parallelism: GPipe to 1F1B

The most visual lab. The deliverable is a schedule diagram, and by the end you should be
able to look at one and read the bubble off it.

Independent of Labs 1 and 2. It depends only on Lab 0 and `common/`.

---

## 1. Objectives

1. Partition the reference model by depth across ranks and pass activations with
   point-to-point operations.
2. Implement micro-batching with an all-forward-all-backward schedule (GPipe/AFAB), then
   1F1B.
3. Derive the bubble fraction and the activation-memory difference, and confirm both by
   measurement.
4. Diagnose and fix pipeline deadlocks.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | Bubble arithmetic and schedule simulation | 45 min |
| 1 | Partition, point-to-point, AFAB, 1F1B | 5–8 h |
| 2 | Real timings on an optional two-GPU host | about 30 min |

Expected Tier-1 run time: about 2 minutes.

Everything here runs under an explicit timeout, because the characteristic failure of
pipeline code is a hang rather than an error.

## 3. Required reading

About 25 pages.

- **Narayanan et al., *PTD-P* §2.2** (arXiv 2104.04473). The schedules and the bubble
  arithmetic. Six pages and it is the core.
- **Huang et al., *GPipe* §2** (arXiv 1811.06965). Micro-batching and the original bubble
  analysis. Four pages.
- **Ultra-Scale Playbook, the pipeline chapter.** AFAB against 1F1B with animations,
  which help more than static figures do for this material.

Reference while coding: `torch.distributed.pipelining` docs and `picotron/pipeline_parallel.py`.

Optional: PipeDream (arXiv 1806.03377) for asynchronous pipelining and weight stashing,
and PTD-P §2.2.2 for the interleaved schedule.

If pipeline scheduling is not clicking on paper, the Picotron video series walks through
building this component step by step. It is the hardest of the three parallelism axes to
read about and the easiest to watch.

## 4. The math

[THEORY.md](THEORY.md). With `p` stages and `m` micro-batches:

**GPipe/AFAB bubble overhead** is `(p-1)/m` relative to ideal work. As a fraction of
elapsed pipeline time it is `(p-1)/(m+p-1)`. Derive both by counting idle slots and
keep their denominators explicit. Peak activation memory holds up to `m`
micro-batches' worth.

**1F1B** has the **same** bubble as AFAB and bounds in-flight activations to `O(p)`
instead of `O(m)`. That is the reason it is the production default, and it is a memory
win rather than a time win. Derive why the steady state alternates one forward and one
backward.

**Interleaved 1F1B** with `v` model chunks per device: overhead ratio
`(p-1)/(v·m)`, elapsed idle fraction `(p-1)/(v·m+p-1)`, at the cost of `v` times
the point-to-point traffic.

**Load balance.** Total time is set by the slowest stage. Derive why an even split by
*layer count* is usually uneven in *FLOPs*: the first stage carries the embedding and the
last carries the LM head and the loss, and for a vocabulary-heavy small model the last
stage can dominate.

## 5. ⚠ Deadlock is the defining hazard

A naive symmetric "every rank sends, then every rank receives" pattern can deadlock when
progress requires peers to post their receives. Transport buffering may hide the defect
for some payloads, but correct code cannot rely on a threshold. Three fixes, and one of
them is required:

- **Odd/even ordering.** Odd ranks receive before they send. Simple, and needs care at
  odd world sizes.
- **Non-blocking `isend`/`irecv` with explicit `wait()`.** Post both, then wait.
- **`dist.batch_isend_irecv`.** Hand the whole exchange to the backend, which then cannot
  deadlock on ordering. **This is the required implementation for the graded path.**

Also: receivers must know the tensor **shape and dtype in advance**. There is no
negotiation in `recv`. Either fix them in the stage config or send a small metadata
tensor first. Getting this wrong produces a hang, not an error, which is why every
pipeline test in this lab has a timeout.

## 6. Build steps

**Step 1 → `test_00_math.py::test_stage_partition_equivalence`**
`starter/partition.py`. Split the reference model into `p` contiguous stages. Running the
stages in sequence in one process must equal the full model's forward.

**Step 2 → `test_01_semantics.py::test_two_stage_forward_backward`**
`starter/p2p.py` plus a naive two-stage forward and backward, batch size 1, no
micro-batching. Loss matches the reference. This establishes the point-to-point mechanics
before any scheduling exists.

**Step 3 → `test_01_semantics.py::test_afab_matches_reference`**
`starter/schedule_afab.py`. Micro-batching with an all-forward-all-backward schedule.
Accumulate gradients across micro-batches and scale the loss by `1/m`.

**Step 4 → `python tasks.py gantt LAB=3`**
`EventLog` instrumentation. **Measure the elapsed idle fraction off the chart and compare
it to `(p-1)/(m+p-1)`.** Also report the `(p-1)/m` overhead ratio separately.

**Step 5 → `test_01_semantics.py::test_1f1b_matches_afab`**
`starter/schedule_1f1b.py`. Same loss as AFAB, substantially lower peak activation
memory, unchanged bubble fraction.

**Step 6 → `python tasks.py bench LAB=3`**
Sweep `m ∈ {1,2,4,8,16}` for `p ∈ {2,4}` and plot measured bubble against theory.

**Optional Step 7 → `test_01_semantics.py::test_optional_torch_pipelining_api_smoke`**
Map the two hand-built schedules to `torch.distributed.pipelining`'s `ScheduleGPipe`
and `Schedule1F1B`. The smoke test checks that the production API used for this
comparison is present; a version-specific production-API integration is not part of the
graded implementation contract.

## 7. Verification

```bash
python tasks.py test LAB=3
```

## 8. Instrumentation & deliverable

`python tasks.py gantt LAB=3` writes `runs/lab3_pipeline_parallel/`:

- `schedule_afab.png` and `schedule_1f1b.png`, side by side. The headline deliverable of
  the lab, and the clearest picture of what a pipeline actually does.
- `bubble_sweep.png`, measured elapsed idle fraction against
  `(p-1)/(m+p-1)`.

Write `runs/lab3_pipeline_parallel/writeup.md`:

1. Put the two Gantt charts side by side. What is different about them, and what is
   identical? Be precise: one thing changes and one thing does not.
2. Measured idle fraction against `(p-1)/(m+p-1)` for each `(p, m)` you swept. Also
   state the `(p-1)/m` overhead ratio. Where measurement exceeds theory, what accounts
   for the difference?
3. Your stage split was FLOP-imbalanced. Say which stage was slowest, by how much, and
   what a better split would have been.
4. 1F1B bounds in-flight activations to `O(p)` instead of `O(m)`. Give your measured
   numbers for `p=4, m=16` and say how much memory that saved.
5. Interleaved 1F1B cuts the bubble by a factor of `v` at the cost of `v` times the
   point-to-point traffic. Using your Lab 0 α and β, at what `v` does that stop being
   worth it on a 25 GB/s InfiniBand link?

## 9. Reconciliation

```bash
python tasks.py reconcile LAB=3
```

## 10. Stretch goals

- **Interleaved 1F1B.** Each device owns `v` non-contiguous chunks of the model. The
  bookkeeping is genuinely harder and the payoff is a `v`-times smaller bubble.
- **Balanced partitioning.** Split by estimated FLOPs rather than by layer count, and
  measure whether the slowest stage got faster.
- **Compose with data parallelism.** Run two pipelines side by side and all-reduce
  gradients across them. This is Lab 4's territory, arrived at from the other direction.
- **Zero-bubble scheduling.** Recent work splits the backward pass into
  input-gradient and weight-gradient halves and schedules them separately. Read the paper
  and say why that helps.

## 11. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Test times out with no output | Symmetric send-then-receive deadlock | Section 5. Use `batch_isend_irecv`. |
| Test times out on the last micro-batch | Warmup and drain counts do not match | The schedule simulation test in `test_00_math.py` catches this before any tensor moves. |
| `recv` returns garbage | Shape or dtype mismatch between sender and receiver | There is no negotiation. Fix both in the stage config or send metadata first. |
| Loss is `m` times too large or small | Micro-batch loss not scaled by `1/m` | Same arithmetic as Lab 1's gradient accumulation. |
| Only the last stage has gradients | Activations sent without `requires_grad`, so the graph does not cross the boundary | The receiving stage must mark its input as requiring grad and send the input gradient back. |
| 1F1B and AFAB give different losses | Micro-batch ordering differs between the two | Same micro-batches in the same order, only the *schedule* changes. |
| Peak memory identical for both schedules | 1F1B is holding all activations anyway | Check that backward runs early enough to release them. |
| Idle fraction much worse than `(p-1)/(m+p-1)` | Stage imbalance, so the model is right and your split is not | Question 3 in the writeup. |

## 12. Where this lives in production

- **`torch.distributed.pipelining`** — `PipelineStage`, `ScheduleGPipe`, `Schedule1F1B`,
  `ScheduleInterleaved1F1B`. Optional Step 7 maps the hand-built schedules to these;
  `_schedule.py` is where the micro-batch ordering lives and it is readable.
- **`megatron/core/pipeline_parallel/schedules.py`** — `forward_backward_pipelining_without_interleaving`
  is 1F1B and `..._with_interleaving` is the interleaved variant. Long, and the structure
  is exactly what you wrote.
- **DeepSpeed `deepspeed/runtime/pipe/engine.py`** — `PipelineEngine`, with the schedule
  expressed as a command stream.
- **`picotron/pipeline_parallel/`** — under 300 lines, and the best thing to diff your
  version against.
