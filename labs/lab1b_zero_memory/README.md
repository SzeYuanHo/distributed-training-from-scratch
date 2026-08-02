# Lab 1B — ZeRO-1 & the Memory Ledger

Small in code, large in understanding. Best done straight after Lab 1, while the DDP
machinery is still fresh.

The question: you have a model with Ψ parameters. Where does the memory actually go, and
what does each ZeRO stage remove?

---

## 1. Objectives

1. Write the exact memory ledger for mixed-precision AdamW training, and verify it
   against measurement.
2. Implement ZeRO stage 1 (optimizer-state sharding) on top of your Lab 1 DDP.
3. Explain what each ZeRO stage removes and what it costs in communication.
4. Predict whether a given configuration fits in 8 GB before running it.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | The ledger arithmetic | 45 min |
| 1 | ZeRO-1 implementation and equivalence | 2–3 h |
| 2 | Real FSDP on an optional two-GPU host | about 30 min |

Expected Tier-1 run time: under 90 seconds. Peak-memory measurement on CPU is noisy, so
the memory tests use generous tolerances and say so.

## 3. Required reading

About 20 pages.

- **Rajbhandari et al., *ZeRO* (arXiv 1910.02054), §3 and §5.** §3 is the memory ledger.
  §5 defines P_os, P_os+g, P_os+g+p, which are stages 1, 2 and 3. Read §5 twice: the
  communication-volume argument is the part people skip and it is the part that matters.
- **Ultra-Scale Playbook, the memory-budget section.** Same content with the activation
  term included, which the ZeRO paper largely sets aside.

Optional: the FSDP paper (arXiv 2304.11277) for how stage 3 is actually engineered, and
`torch.distributed.optim.ZeroRedundancyOptimizer` source, which is close to what you are
about to write.

## 4. The math

[THEORY.md](THEORY.md). The headline, per parameter, mixed-precision AdamW:

| Component | Bytes |
|---|---|
| bf16/fp16 parameters | 2Ψ |
| bf16/fp16 gradients | 2Ψ |
| fp32 master parameters | 4Ψ |
| fp32 Adam first moment | 4Ψ |
| fp32 Adam second moment | 4Ψ |
| **Total** | **16Ψ** |

Written as `2Ψ + 2Ψ + KΨ` with K = 12 for the optimizer group. Then:

- **ZeRO-1** shards the 12Ψ: `2Ψ + 2Ψ + 12Ψ/N`
- **ZeRO-2** additionally shards gradients: `2Ψ + (2Ψ + 12Ψ)/N`
- **ZeRO-3** additionally shards parameters: `16Ψ/N`

Derive each. Then derive activation memory separately, because the table above ignores it
and at large batch times sequence length it dominates everything in it.

**The communication fact that surprises people, and which a test asserts:** ZeRO-1 and
ZeRO-2 move exactly as many bytes as plain DDP. An all-reduce *is* a reduce-scatter
followed by an all-gather, so replacing one with the other two changes where the bytes go
and not how many there are. ZeRO-3 is genuinely different: it adds parameter all-gathers
in both the forward and the backward pass.

## 5. Build steps

**Step 1 → `test_00_math.py::test_ledger_*`**
`starter/ledger.py`. Reproduce `common/memory.py::analytic_ledger` from the ZeRO paper.

Please do this before reading `common/memory.py`. Nothing enforces it, and the lab is
worth much less if you skip it. The test compares your function against that one, so
copying it makes the test pass and teaches you nothing, which is a trade you can make and
should not.

**Step 2 → `test_00_math.py::test_fits_in_*`**
`predict_fits(config, vram_bytes)`. Given a model, a batch, and a card, will it fit? This
is the skill: not measuring OOM, predicting it.

**Step 3 → `test_01_semantics.py::test_zero1_loss_matches_ddp`**
`starter/zero1.py::ZeRO1Optimizer`. Each rank owns a contiguous shard of parameters and
keeps optimizer state only for that shard. Reduce-scatter the gradients, step on the
local shard, all-gather the updated parameters.

**Step 4 → `test_01_semantics.py::test_zero1_communication_equals_ddp`**
Assert the surprising fact. Total bytes for ZeRO-1 should equal DDP's.

**Step 5 → `test_01_semantics.py::test_optimizer_state_shrinks_with_world_size`**
Measure the optimizer state and confirm it falls as 1/N.

**Step 6 → the deliverable**
`python tasks.py bench LAB=1b` produces peak memory against N, analytic and measured,
with the gap attributed.

## 6. Verification

```bash
python tasks.py test LAB=1b
```

## 7. Instrumentation & deliverable

`runs/lab1b_zero_memory/memory_curve.png`: per-rank memory against world size, one line
per ZeRO stage, analytic as dashed and measured as solid.

Write `runs/lab1b_zero_memory/writeup.md`:

1. Your analytic ledger for `small` at N=1 against the measured peak. What is the gap in
   bytes, and what fraction of it can you attribute to activations, to the autograd
   graph, and to the allocator?
2. ZeRO-1 moves the same total bytes as DDP. Prove it in one paragraph, then say why the
   *latency* profile still differs.
3. At what N does ZeRO-1 stop helping for the `small` preset, and why? (Hint: look at
   which term stops shrinking.)
4. On an 8 GB GPU, what is the largest model you can train with plain AdamW at batch 8,
   sequence 512? With ZeRO-1 at N=4? Show the arithmetic.
5. ZeRO-3 shards parameters and therefore has to all-gather them for every forward and
   every backward. Under what interconnect and model-size conditions is that a good
   trade?

## 8. Reconciliation

```bash
python tasks.py reconcile LAB=1b
```

## 9. Stretch goals

- **ZeRO-2.** Shard gradients too. A small change to where the reduce-scatter output
  lands, and a real change to the memory curve.
- **Real FSDP at Tier 2 (optional, not graded).** Run
  `torch.distributed.fsdp.FullyShardedDataParallel` on the same model on 2xT4 and
  compare its measured memory against your stage-3 prediction. The published answer
  set does not claim this hardware-dependent extension was run.
- **Activation checkpointing.** Add `torch.utils.checkpoint` to the reference model's
  blocks and measure the activation-memory reduction against the extra compute. The
  Korthikanti formula in `common/memory.py::activation_estimate` gives you the prediction.
- **Calibrate the headroom.** `memory.fits_in` assumes 15% allocator headroom. On
  hardware you control, measure the actual requirement by bisecting batch size until OOM.

## 10. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Measured memory far below prediction | Adam allocates `m` and `v` lazily on the first `step()` | Measure after step 1, not step 0. |
| Measured memory far above prediction | Activations, which the ledger excludes | Pass `activations=` or account for it in the writeup. |
| ZeRO-1 loss diverges from DDP after a few steps | Parameters all-gathered before the local step, not after | Order is reduce-scatter, step, all-gather. |
| ZeRO-1 uses *more* memory than DDP | Keeping full optimizer state and also a shard | The point is to allocate state only for your shard. |
| Peak memory on CPU is meaningless | `getrusage` measures the whole process | Compare trends, not absolutes. The tests do. |
| Shard boundaries split a parameter | Contiguous byte sharding cuts tensors in half | Shard by parameter, or handle the split explicitly. Say which you chose. |

## 11. Where this lives in production

- **`torch.distributed.optim.ZeroRedundancyOptimizer`** — stage 1, close to what you
  wrote. Read `_partition_parameters`.
- **DeepSpeed `deepspeed/runtime/zero/stage_1_and_2.py`** — the original. Large, and the
  bucket and offload machinery around the core idea is instructive.
- **`torch.distributed.fsdp`** — stage 3, plus prefetching, plus mixed precision, plus
  CPU offload. `_runtime_utils.py` is where the all-gather scheduling lives.
- **`picotron/data_parallel/`** — the readable version.
