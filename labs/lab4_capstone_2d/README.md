# Lab 4 — Capstone: 2D Composition & Scaling Report

Where the separate pieces become judgement.

Labs 0 to 3 each answered "how does this work". This one answers "which of these should I
use, and why", which is the question anyone will actually ask you.

The deliverable is a two-to-three page scaling report. It is the single best artefact in
this repository to show someone.

---

## 1. Objectives

1. Compose data parallelism with tensor parallelism over a `DeviceMesh`, with correct
   process-group construction.
2. Use the cost and memory models from Labs 0 to 3 to *choose* a configuration before
   running it.
3. Write a short scaling report defending the choice with numbers.

## 2. Time & tier budget

| Tier | What | Budget |
|---|---|---|
| 0 | Planner scoring against hand-computed values | 1 h |
| 1 | 2D composition on 4 gloo ranks | 3–4 h |
| 2 | Validate on an optional two-GPU host and compute MFU | about 45 min |

Expected Tier-1 run time: about 2 minutes. Needs at least 4 ranks for a `2 x 2` mesh, so
a machine with fewer than 4 cores will be slow.

## 3. Required reading

About 25 pages.

- **Scaling Book, the parallelism-selection chapter.** Free at
  https://jax-ml.github.io/scaling-book/. The whole point of that book is producing the
  estimate you are about to produce.
- **PTD-P §3 and its empirical guidelines** (arXiv 2104.04473). The rules of thumb come
  from here and they are backed by measurements on thousands of GPUs.
- **Ultra-Scale Playbook, "finding the best configuration".**

Reference: `pytorch/torchtitan` configs, which are a production answer to the same
question for real models on real clusters.

## 4. The math

You have already derived everything. This lab assembles it:

| From | What you get |
|---|---|
| Lab 0 | α and β, and the ring all-reduce cost model |
| Lab 1 | overlap, and the exposed-communication expression |
| Lab 1B | the memory ledger and the ZeRO stages |
| Lab 2 | TP communication volume and the scaling ceiling |
| Lab 3 | the pipeline bubble and stage imbalance |

The new content is **MFU**. Model FLOPs per token is about `6Ψ` for a forward and backward
pass. Multiply by tokens per second, divide by the device's peak FLOP/s.

**HFU** counts every FLOP the hardware executed, including recomputation. **MFU** counts
only the FLOPs the model definition requires. With activation recomputation on, HFU
exceeds MFU, and the gap is the price you paid in compute to save memory. Quoting the
wrong one flatters your number by 30% or more, so know which you are reporting.

## 5. ⚠ Rank-to-mesh mapping

`world_size = dp × tp`, and how you assign ranks to mesh coordinates decides which traffic
crosses which link.

**Tensor-parallel groups must be the innermost, fastest dimension.** On an 8-GPU NVLink
node with `dp=2, tp=4`, you want ranks 0-3 forming one TP group and 4-7 the other, so
every TP all-reduce stays on NVLink. Get it backwards and your TP groups span the slow
inter-node link, which is the difference between "works" and "works at a third the speed".

Lab 2's ceiling derivation is why. TP communicates the most and tolerates latency the
least, so it gets the best link. Data parallelism communicates once per step and overlaps
with the backward pass, so it can live on the worst one.

`starter/mesh.py` makes you build the groups by hand, then cross-check against
`torch.distributed.device_mesh.init_device_mesh`. The test asserts the coordinate mapping,
because "it ran" is not evidence you got this right.

## 6. Build steps

**Step 1 → `test_01_semantics.py::test_mesh_coordinates`**
`starter/mesh.py`. Build the `dp × tp` mesh, construct both process groups, verify each
rank's coordinates. Print the mapping and explain it in the writeup.

**Step 2 → `test_01_semantics.py::test_2d_composition_matches_reference`**
`starter/compose.py`. Data-parallel gradient averaging over the DP group, tensor-parallel
layers within the TP group. Loss matches a single-process reference; collective counts
match your prediction on both groups.

**Step 3 → `test_00_math.py::test_planner_scores`**
`starter/planner.py`. Enumerate candidate configurations and score each on predicted
memory per GPU, predicted communication volume per step, and predicted bubble or overhead.
Derive the local micro-batch as
`global_batch_tokens / (dp × micro_batches × sequence_length)` complete sequences;
non-integral candidates are infeasible rather than rounded. Rank the valid candidates.

**This is the lab.** Everything else is plumbing around it.

**Step 4 → `python tasks.py bench LAB=4`**
Compute MFU for a Tier-2 run and explain why MFU and HFU differ under recomputation.

**Step 5 → the report**
Compare your reasoning against torchtitan's published configurations and the PTD-P
guidelines: roughly, tensor parallelism within a node, pipeline parallelism across nodes,
data parallelism outermost.

## 7. Verification

```bash
python tasks.py test LAB=4
```

## 8. Instrumentation & deliverable

**`runs/lab4_capstone_2d/scaling_report.md`**, two to three pages. The template is in
[SOLUTION.md](SOLUTION.md), and it should contain:

1. **The problem.** A model size, a cluster, a target. Use one of the profiles in
   `common/cluster_specs.py` or invent one and state its numbers.
2. **The configuration table.** Every candidate `(dp, tp, pp, micro-batches, ZeRO stage)`
   with its derived sequences per micro-batch, predicted memory per GPU, predicted
   communication bytes per step, predicted bubble, and a feasibility verdict. Ten to
   twenty rows.
3. **Your choice, with the argument.** Which one and why, in numbers.
4. **The measurement.** What actually happened when you ran the top candidate.
5. **The reconciliation.** Where the prediction missed and what the model left out.

Also `runs/lab4_capstone_2d/mfu.png`: achieved MFU against configuration.

## 9. Reconciliation

```bash
python tasks.py reconcile LAB=4
```

## 10. Stretch goals

- **3D.** Add pipeline parallelism from Lab 3 for `dp × tp × pp`. The mesh construction
  generalises; the schedule interaction is where it gets interesting.
- **Automatic search.** Have the planner enumerate the whole space and return the
  Pareto front over throughput and memory.
- **Validate against torchtitan.** Take a published config for a real model on real
  hardware, feed the same parameters to your planner, and see whether it picks the same
  answer. When it does not, work out who is wrong.
- **Add a cost axis.** Cloud GPUs have prices. Rank configurations by dollars per token
  rather than by tokens per second, and watch the answer change.

## 11. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| TP traffic is slow on a real cluster | TP groups span the slow link | Section 5. TP is innermost. |
| Loss matches at `dp=1` and not at `dp=2` | Gradients averaged over the wrong group | DP group only. The TP group already agrees. |
| Collective counts are double what you predicted | Both groups all-reducing the same tensor | Each parameter is either replicated or sharded, not both. |
| The planner recommends a configuration that OOMs | Activations left out of the memory model | Lab 1B's `activation_estimate`. |
| Changing `m` never changes activation memory | The local micro-batch was hardcoded | Derive complete sequences from `global_batch_tokens / (dp × m × seq_len)`. |
| A bf16 row appears executable on a T4 | The planner ignored dtype capability | Reject bf16 when the cluster profile reports no bf16 support; use fp16 for T4. |
| MFU above 100% | Peak FLOPs wrong, or you counted recomputation | You computed HFU. See section 4. |
| Every candidate scores identically | The model is not sensitive to what you varied | Check the cost function actually depends on `tp`. |

## 12. Where this lives in production

- **`pytorch/torchtitan`** — `train_configs/*.toml` for the answers and
  `torchtitan/parallelisms/parallel_dims.py` for how the mesh is built. Closest production
  reference to what you wrote.
- **`torch.distributed.device_mesh`** — `init_device_mesh` and `DeviceMesh`, the API step 1
  cross-checks against. `_mesh_resources` shows how sub-meshes are derived.
- **`megatron/core/parallel_state.py`** — `initialize_model_parallel`, which builds all
  the groups for DP, TP, PP and context parallelism. Read the rank-arithmetic comments;
  they are the section-5 argument in code.
- **PTD-P §3** for the empirical guidelines your report should agree or disagree with.
