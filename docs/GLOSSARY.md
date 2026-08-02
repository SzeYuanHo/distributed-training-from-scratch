# Glossary

Terms as this repository uses them, with the symbol each lab uses. Where a term is
genuinely ambiguous in the literature, that is noted, because those are the ones that
cause arguments.

---

## Processes and groups

**Rank** — a process's global index, `0` to `world_size - 1`. `dist_utils.rank()`. Not a
GPU index, though the two often coincide.

**Local rank** — index within one node. On a single-node job it equals the rank. On four
nodes of eight GPUs, rank 11 has local rank 3. `LOCAL_RANK`, used to pick a CUDA device.

**World size** — total number of processes. `N` in the cost models.

**Process group** — a subset of ranks that can talk to each other. The default group is
everyone. Lab 4 builds two subgroups so data-parallel and tensor-parallel traffic use
different links.

**Backend** — the transport. `gloo` is CPU-oriented and is the portable path when the
installed Torch build provides a usable transport; `doctor` probes it because some
Windows CUDA wheels do not. `nccl` targets NVIDIA GPUs and is normally much faster.
Use one process per device: shared-device ranks are unsupported for this workflow and
can hang or report invalid usage.

**Device mesh** — an N-dimensional arrangement of ranks. A `2 x 4` mesh over 8 ranks gives
each rank a `(dp, tp)` coordinate and defines both groups at once.
`torch.distributed.device_mesh`.

---

## Collectives

**Collective** — an operation every rank in a group participates in. If one rank does not
arrive, everyone waits forever, which is the source of most hangs.

**Broadcast** — one rank's tensor overwrites everyone's.

**Reduce** — combine every rank's tensor onto one rank. Only that rank's result is
meaningful afterwards, and the others' contents are unspecified.

**All-reduce** — reduce, and every rank gets the result. The workhorse of data
parallelism.

**Scatter** — one rank distributes different pieces to each rank.

**Gather** — the inverse. **All-gather** gathers onto every rank.

**Reduce-scatter** — reduce, then scatter, so rank `i` keeps slice `i` of the total. Half
of an all-reduce, and the basis of ZeRO.

**All-to-all** — a distributed transpose. Rank `i`'s chunk `j` goes to rank `j`'s slot
`i`. The most expensive standard collective, and the one mixture-of-experts routing needs.

**Point-to-point** — `send` and `recv` between two ranks. Pipeline parallelism is built
from these.

**Ring algorithm** — arrange ranks in a circle, each talking only to its neighbour.
Moves `2(N-1)/N · M` bytes per rank, which approaches `2M` and never exceeds it.
Bandwidth-optimal. Lab 0.

**Tree algorithm** — a logarithmic-depth pattern. `2·log₂N` latency hops instead of the
ring's `2(N-1)`. It is latency-efficient and can win on small messages. NCCL may select
tree, ring or other mechanisms at runtime based on message size, topology, build and
configuration; inspect runtime logs rather than assuming a universal crossover.

---

## Cost models

**α (alpha)** — per-message latency, seconds. Grama writes `t_s`. Independent of message
size.

**β (beta)** — inverse bandwidth, seconds per byte. Grama writes `t_w` per word.

**α–β model** — `T = α + β·m` for a message of `m` bytes. The straight line every cost
derivation in this repo starts from.

**Algorithm bandwidth** — `M / T`. Falls as the rank count rises even on perfect
hardware.

**Bus bandwidth** — `2(N-1)/N · M / T` for an all-reduce. Scaled so a perfect flat ring
reports roughly the hardware's link bandwidth regardless of N. NVIDIA's `nccl-tests`
reports both algorithm bandwidth and this derived `busbw`; the latter is a correction
factor, not a literal wire-rate measurement for every hierarchical or hardware-assisted
algorithm. Knowing which number was quoted prevents a lot of confusion.

**Roofline** — the model that says a kernel is limited either by compute or by memory
bandwidth, whichever binds first. Scaling Book chapter 1.

**Arithmetic intensity** — FLOPs per byte moved. Where a kernel sits on the roofline.

---

## Parallelism axes

**Data parallelism (DP)** — every rank holds the whole model and a different slice of the
batch. Gradients are all-reduced. Simplest, and memory-bound at scale.

**Tensor parallelism (TP)** — split individual weight matrices across ranks. The logical
all-reduce payload shape is independent of TP degree while per-rank compute falls as
`1/t`. Ring link traffic additionally carries the `2(t-1)/t` factor, so it is bounded
rather than exactly constant. Either way the compute-to-communication ratio deteriorates
with `t`, which is why TP favours a fast interconnect domain. Lab 2.

**Pipeline parallelism (PP)** — split the model by depth, one contiguous block of layers
per rank. Cheap communication, at the cost of a bubble. Lab 3.

**Sequence parallelism** — shard along the sequence dimension in the regions where tensor
parallelism leaves activations replicated. Same total bytes, lower activation memory.
Lab 2 stretch goal.

**Context parallelism** — shard the sequence for attention itself, which needs a ring
exchange of keys and values. Ring Attention. Out of scope here; only earns its keep at
long sequence lengths.

**Expert parallelism** — distribute mixture-of-experts experts across ranks, with
all-to-all routing. Out of scope.

**3D parallelism** — DP times TP times PP. Lab 4 does the 2D case.

---

## Data parallelism internals

**Bucket** — a group of gradients all-reduced as one flat buffer. One 4 MiB collective
costs one latency; a thousand 4 KB ones cost a thousand. Lab 1.

**Bucket size** — bytes per bucket. PyTorch defaults to 25 MiB. Larger means fewer
latencies and a longer un-overlappable tail.

**Reverse registration order** — bucketing parameters back-to-front, because gradients
arrive in reverse order of the forward pass. Makes bucket 0 fill first so its collective
launches earliest.

**Overlap** — running communication concurrently with the backward pass that produces it.
The reason DDP is fast.

**Exposed communication** — communication time with no compute running underneath. Never
zero: the last bucket cannot start until the last gradient exists.

**Gradient accumulation** — running `k` micro-batches before stepping, so the effective
batch is `k` times larger without `k` times the memory. `no_sync` suppresses the
collective on all but the last.

**Effective batch size** — `per_rank_batch × world_size × accumulation_steps`. What the
learning rate should be scaled against.

**Linear scaling rule** — multiply the batch by `k`, multiply the learning rate by `k`.
Goyal et al. Holds until it does not.

---

## Memory

**Memory ledger** — the accounting of where per-rank bytes go: parameters, gradients,
master weights, optimizer state, activations.

**16Ψ** — bytes per parameter for mixed-precision AdamW: `2 + 2 + 4 + 4 + 4`. The number
to have memorised. `Ψ` is the parameter count.

**Master weights** — the fp32 copy the optimizer updates. Needed because a typical bf16
update is too small to change a bf16 parameter. Belongs to the optimizer group, which is
why ZeRO-1 shards it.

**Activation memory** — intermediate tensors held for the backward pass. Absent from the
16Ψ table and frequently larger than everything in it.

**ZeRO** — sharding the redundant state across data-parallel ranks.
**Stage 1** shards optimizer state, **stage 2** adds gradients, **stage 3** adds
parameters. Stages 1 and 2 move exactly as many bytes as plain DDP; stage 3 does not.

**FSDP** — PyTorch's implementation of ZeRO-3 with prefetching and mixed precision.

**Activation recomputation** — discard activations in the forward pass and recompute them
in the backward. Trades compute for memory. **Selective** recomputation drops only the
quadratic attention term, which buys most of the saving for a fraction of the cost.

---

## Pipeline scheduling

**Micro-batch** — a slice of the batch that flows through the pipeline independently. `m`
of them.

**Bubble** — idle time while the pipeline fills and drains. For `p` stages and `m`
micro-batches, `(p-1)/m` is the overhead relative to ideal work; the fraction of
elapsed pipeline time is `(p-1)/(m+p-1)`.

**GPipe / AFAB** — all-forward-all-backward. Every micro-batch's forward, then every
backward. Simple, and holds `O(m)` micro-batches' activations.

**1F1B** — one forward, one backward, alternating after a warmup. **Same** bubble as AFAB,
activations bounded by `O(p)`. The production default, and the win is memory, not time.

**Interleaved 1F1B** — `v` non-contiguous chunks per device. Overhead ratio
`(p-1)/(v·m)`, elapsed idle fraction `(p-1)/(v·m+p-1)`, at `v` times the
point-to-point traffic.

**Warmup / steady state / drain** — the three phases of 1F1B. In this repository,
the warmup count means forwards completed before the first backward:
`min(p-i, m)` for stage `i`. The number of forward-only steps before the first
paired 1F1B iteration is one smaller when steady state exists.

**Stage imbalance** — total time is set by the slowest stage. An even split by layer count
is usually uneven in FLOPs, because the first stage carries the embedding and the last
carries the LM head.

---

## Tensor parallelism internals

**Column parallel** — split a weight along its output dimension. Output is sharded; the
bias is sharded too. No forward communication.

**Row parallel** — split along the input dimension. Each rank produces a partial sum, so
the forward needs an all-reduce. **The bias is not sharded and is added after the
reduction.**

**f and g** — the conjugate operators. `f` is identity forward and all-reduce backward;
`g` is all-reduce forward and identity backward. Swapping them leaves the forward correct
and the gradients wrong.

**Vocab parallel** — shard the embedding and the LM head along the vocabulary dimension.
The loss is then computed without ever materialising the `(b, s, V)` logit tensor.

**TP scaling ceiling** — the degree at which communication time exceeds compute time.
Typically 8 on NVLink, low single digits over PCIe. The reason TP does not cross node
boundaries.

---

## Efficiency

**MFU (model FLOPs utilisation)** — FLOPs the *model definition* requires, divided by the
device's peak. `6Ψ` per token for a forward and backward pass.

**HFU (hardware FLOPs utilisation)** — every FLOP the hardware actually executed,
including recomputation. **HFU exceeds MFU when recomputation is on**, and quoting the
wrong one flatters your number by a third.

**Scaling efficiency** — throughput on `N` devices divided by `N` times the throughput on
one. Ideal linear scaling is 1; real measurements may be below it, or occasionally above
it because of noise, cache effects, or a changed effective workload.

**Strong scaling** — fixed problem, more devices. **Weak scaling** — problem grows with
the devices.

**Amdahl's law** — speedup is capped by the serial fraction. **Gustafson's law** — the
optimistic counterpart, where the problem grows to fill the machine.

---

## This repository

**Tier 0** — one process, no `torch.distributed`. Pure maths.

**Tier 1** — several CPU processes on gloo. Real collectives, real correctness. The
default development loop.

**Tier 2** — two or more CUDA devices with NCCL. The only tier where a timing claim is
honest.

**Reference implementation** — the frozen single-process ground truth each lab compares
against. `common/reference_gpt.py` is the model; each lab has its own for the specific
comparison.

**Negative test** — a test that passes by demonstrating a failure mode, so you can see
that the suite would catch it.

**Predict, measure, reconcile** — write your numeric prediction in `predictions.yaml`
before running, measure it, and print both. Being wrong is the learning event.
