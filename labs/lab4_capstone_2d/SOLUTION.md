# Lab 4 — Solution notes

---

## The mesh

The rank arithmetic is three lines and the trap is elsewhere:

```python
tp_rank = global_rank % tp_size
dp_rank = global_rank // tp_size
```

**Every rank must call `dist.new_group` for every group.** `new_group` is collective: all
ranks participate in creating each group, even the ones that will not belong to it. Build
only your own groups and the job deadlocks on the first `new_group` call, before any of
your code runs, and the hang looks like a bug in the model.

```python
tp_groups = []
for i in range(dp_size):
    ranks = list(range(i * tp_size, (i + 1) * tp_size))
    group = dist.new_group(ranks)  # every rank calls this
    if global_rank in ranks:
        my_tp_group = group  # only some keep it
```

Same pattern for the DP groups. It looks wasteful and it is what the API requires.

Torch's `init_device_mesh("cpu", (dp, tp), mesh_dim_names=("dp", "tp"))` uses the same
convention: the last dimension varies fastest, so TP ends up innermost. Confirming that
rather than assuming it is worth five minutes, because it is exactly the kind of thing
that changes between versions and produces a slow job rather than a broken one.

---

## Composition

`sync_gradients` is one loop with one thing to get right:

```python
if layout.dp_size == 1:
    return
for p in self.parameters():
    if p.grad is None:
        continue
    collectives.all_reduce(p.grad, group=layout.dp_group)
    p.grad /= layout.dp_size
```

`group=layout.dp_group`, not `group=None`. Passing `None` averages over the world, which
divides by `dp * tp` when you meant `dp`, and the effective learning rate comes out `tp`
times too small. The loss falls, slowly, and it looks like a hyperparameter problem.

The data-loading rule is the other half:

```python
x, y = get_batch(..., rank=layout.dp_rank, world_size=layout.dp_size, ...)
```

Every rank in a TP group must see the *same* batch, because they are jointly computing one
forward pass over it. Indexing by global rank gives each TP rank a different batch and the
tensor-parallel arithmetic then combines activations from unrelated examples. Nothing
raises.

---

## The planner

Take the time to build this properly. It is the part of the lab worth showing someone.

**Structure it as a pure function.** `score(model, config, cluster) -> ScoredConfig`, with
no I/O and no globals. Then the tests are trivial, you can enumerate thousands of
candidates in a second, and you can swap the cluster and re-rank without touching
anything else.

**Memory, term by term:**

```python
params_per_gpu = model.n_params // (config.tp * config.pp)
ledger = memory.analytic_ledger(
    params_per_gpu,
    model.dtype,
    model.optimizer,
    config.zero_stage,
    config.dp,  # ZeRO shards across dp only
)
```

The `world_size=config.dp` is the part to be careful about. ZeRO shards optimizer state
across the data-parallel dimension. Tensor and pipeline parallelism have already reduced
the parameter count each rank is responsible for, and passing the full world size here
double-counts the saving.

Activations:

```python
micro_sequences, micro_tokens = model.micro_batch_shape(config)
act = memory.activation_estimate(
    micro_sequences,
    model.seq_len,
    model.hidden,
    model.n_layers // config.pp,  # each stage holds its own layers
    model.n_heads,
    model.dtype,
    recompute=model.recompute,
    tensor_parallel=config.tp,
)
inflight = config.pp if config.schedule == "1f1b" else config.micro_batches
act_per_gpu = act * min(inflight, config.micro_batches)
```

Here `micro_sequences = global_batch_tokens / (dp * m * seq_len)`. All quantities
must divide exactly: TP and PP cooperate on the same data and therefore do not reduce
the local micro-batch. Rejecting a non-integral candidate is safer than rounding it and
quietly changing the optimizer-step batch.

That `inflight` line is Lab 3's result doing work in a different lab, and it is the single
place where the schedule choice changes the memory answer.

**Communication, three independent terms.** Compute each separately and keep them
separate in the output. When a configuration looks wrong, "which axis is generating the
traffic" is the first question, and a single total cannot answer it.

**Feasibility with a reason.** `"needs 94.2 GiB, 68 GiB usable after headroom, short by
26.2 GiB"` is actionable. `False` is not.

**The estimated step time is rough and that is fine.** You are ranking candidates. A model
that gets the ordering right and the absolute numbers wrong by 40% is useful; one that is
accurate to 5% for one configuration and untested elsewhere is not.

---

## The scaling report

The deliverable, and the artefact from this repository most worth showing to a hiring
engineer. Two to three pages. Structure it like this:

**1. The problem.** One paragraph. Model size, cluster, target throughput or wall-clock
budget. Cite where the hardware numbers came from; `common/cluster_specs.py` has sources
for each.

**2. The candidates.** The table from `report_table`, trimmed to the interesting rows.
Fifteen is plenty. Columns: configuration, memory per GPU, communication per step split by
axis, bubble, estimated step time, feasible.

**3. The recommendation.** Which one, and why, in numbers. Not "tp=4 balances memory and
communication" but "tp=4 and eight accumulation micro-batches leave 16 complete
sequences per local micro-batch; with recomputation the model fits inside the 85% VRAM
budget, while keeping TP traffic on NVLink."

**4. What actually happened.** Run the top candidate at whatever scale you can reach. An
available T4 pair is enough to check ordering for an fp16 T4-compatible problem, even if
the absolute numbers are not.

**5. The reconciliation.** Where the prediction missed and what the model left out. This
is the most valuable section, and it is the one people skip. If your predicted step time
was 47 ms and you measured 71, the interesting content is the 24 ms, not the apology.

**6. What would change the answer.** One paragraph. If the interconnect were 25 GB/s
instead of 300, if the sequence were 32k instead of 4k, if you had 64 GPUs instead of 8.
Showing that you know which way the answer moves is more convincing than the answer.

---

## Two things worth getting right in the report

**Do not quote HFU as MFU.** With activation recomputation on, the hardware executes an
extra forward pass per layer, so HFU exceeds MFU by roughly a third. Quoting HFU makes your
run look better and is the wrong number for "how efficiently am I using this machine to
train this model". Say which you computed.

**Be honest about what you measured against what you derived.** The T4-pair and A100
numbers in the report are derived from published peaks and your own cost models. You did
not run an eight-GPU job. Saying so costs you nothing and lying about it costs you
everything the first time someone asks a follow-up question.

The derivation is the skill. Producing a defensible configuration recommendation before
requesting cluster time is exactly the thing that gets asked, and "I do not have the
hardware so I modelled it, and here is the model" is a strong answer.
