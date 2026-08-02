# Curriculum and reading map

This repository is a lab manual **for** a small set of texts, not a replacement for them.
This file says which texts, and exactly which parts to read for each lab.

The hard rule: **20 to 40 pages of required reading per lab.** More than that and it stops
being read.

---

## 1. The three anchors

Pick these three and stay with them. A long bibliography is a way of avoiding reading
anything.

### A. The Ultra-Scale Playbook

*Training LLMs on GPU Clusters*, Hugging Face nanotron team, 2025.
<https://huggingface.co/spaces/nanotron/ultrascale-playbook>

**The main course text.** Free, around 150 pages, and the closest match to this curriculum
that exists. It covers data parallelism with gradient-synchronisation optimisation, ZeRO
stages 1 to 3, tensor and sequence parallelism, context parallelism with Ring Attention,
pipeline scheduling with AFAB and 1F1B, activation recomputation, and memory budgeting
across weights, gradients, optimizer states and activations. All of it grounded in over
4,000 scaling experiments on up to 512 GPUs. It also introduces HFU and MFU, which Lab 4
uses.

A modern web-book rather than a printed classic. On this specific subject it is the state
of the art, it is free, and its structure maps almost one-to-one onto these labs. Use the
printed classics below for the timeless parallel-computing theory it assumes.

Companion code from the same team, both worth having open:

- **[picotron](https://github.com/huggingface/picotron)** — a minimalist 4D-parallelism
  framework for education. `train.py`, `model.py`, and each
  `[data|tensor|pipeline|context]_parallel.py` are under 300 lines apiece. It has a CPU
  mode, which is the same technique this repo's Tier 1 uses. If you are solving the labs,
  read it after your own attempt; if you are studying completed systems, use it alongside
  the published `answers/`.
- **[nanotron](https://github.com/huggingface/nanotron)** — the production version. Worth
  a look to see what the real thing costs in complexity.
- Ferdinand Mom's step-by-step video series on building picotron, linked from its README.
  Useful if you prefer video for the pipeline-scheduling material, which is the hardest of
  the three axes to read about on paper.

### B. How to Scale Your Model

*A Systems View of LLMs*, Austin, Douglas, Frostig, Levskaya et al., Google DeepMind, 2025.
<https://jax-ml.github.io/scaling-book/>

**The quantitative reasoning text.** Free. This is where you learn to *predict* rather
than measure: roofline analysis, what limits scaling, the arithmetic of collectives, and
how to choose a parallelism scheme for given hardware. Its stated goal is that by the end
you can estimate the best scheme for a transformer on given hardware and roughly how long
training should take. That is exactly the Lab 4 capstone skill.

Written around TPUs, and the reasoning transfers directly. It now includes a section
specifically on NVIDIA GPUs. Chapter 1 (Rooflines) and the training and parallelism
chapters are the required parts.

### C. Introduction to Parallel Computing

Grama, Gupta, Karypis & Kumar, 2nd edition, Addison-Wesley.

**The printed classic underneath everything.** Nothing in modern distributed training is
new here. It is the same collectives and the same cost models applied to a new workload.

- **Ch. 2 §2.5** — the communication cost model, `t_s + t_w·m`, which is the α–β model
  everyone else uses with different letters.
- **Ch. 3** — decomposition, task and data mapping, load balancing. Directly underlies
  pipeline stage balancing in Lab 3.
- **Ch. 4, "Basic Communication Operations"** — **the single most valuable chapter for
  this project.** Broadcast, all-to-all, all-reduce, prefix sums, scatter and gather, with
  full cost derivations on ring and hypercube topologies. Lab 0 is essentially chapter 4
  implemented in PyTorch.
- **Ch. 5 §5.1–5.4** — speedup, efficiency, scalability, isoefficiency, Amdahl and
  Gustafson. The vocabulary for Lab 4.

**Free alternative:** Victor Eijkhout, *The Art of HPC, Vol. 2: Parallel Programming for
Science and Engineering*, at <https://theartofhpc.com>. Covers the same collectives and
cost-model material.

---

## 2. Papers, by section

Short and directly implementable. Each lab needs one or two, and you read the named
section rather than the paper.

| Paper | Read | For |
|---|---|---|
| **PyTorch Distributed** — Li et al., VLDB 2020 (arXiv 2006.15704) | §3: gradient bucketing, overlap, the reverse-order argument | **Lab 1. This is the blueprint.** |
| **ZeRO** — Rajbhandari et al., SC'20 (arXiv 1910.02054) | §3 memory ledger, §5 the three stages | **Lab 1B** |
| **Accurate, Large Minibatch SGD** — Goyal et al. 2017 (arXiv 1706.02677) | §2: linear scaling rule, gradual warmup | **Lab 1** |
| **Megatron-LM** — Shoeybi et al. 2019 (arXiv 1909.08053) | §3 entirely: MLP and attention partitioning, the f/g operators | **Lab 2. The blueprint.** |
| **Reducing Activation Recomputation** — Korthikanti et al. 2022 (arXiv 2205.05198) | §3 sequence parallelism, §4 the activation-memory formula | **Lab 2 stretch, Lab 1B** |
| **GPipe** — Huang et al. 2018 (arXiv 1811.06965) | §2: micro-batching and the bubble | **Lab 3** |
| **PTD-P** — Narayanan et al., SC'21 (arXiv 2104.04473) | §2.2 pipeline schedules and bubble arithmetic; §3 composing DP/TP/PP | **Labs 3 and 4. The blueprint for both.** |
| **Horovod** — Sergeev & Del Balso 2018 (arXiv 1802.05799) | the ring all-reduce description | **Lab 0** |

Full citations in [`references.bib`](../references.bib). No PDFs are vendored here; links
only.

---

## 3. Per-lab reading map

### Lab 0 — Collectives & Cost Models

**Required (~30 pages):** Grama §2.5 (the cost model), Ch. 4 §4.1–4.3
(broadcast, reduce, all-to-all) and §4.7 (all-reduce, prefix sums). Scaling Book Ch. 1
(Rooflines).

**Reference while coding:** the `torch.distributed` collective-semantics table. Keep it
open; the difference between `all_gather` and `all_gather_into_tensor` costs twenty
minutes otherwise.

**Optional:** the Horovod paper. NCCL developer docs on runtime algorithm and protocol
selection, including ring and tree.

### Lab 1 — DDP from Scratch

**Required (~25 pages):** PyTorch DDP paper §3, which is the whole lab. Ultra-Scale
Playbook data-parallelism chapter. Goyal et al. §2.

**Reference:** `torch/nn/parallel/distributed.py`, and the `register_hook` family in the
autograd docs.

**Optional:** the PyTorch DDP design note. `reducer.cpp`, which becomes readable *after*
you finish.

### Lab 1B — ZeRO-1 & the Memory Ledger

**Required (~20 pages):** ZeRO paper §3 and §5. Ultra-Scale Playbook memory-budget
section.

**Optional:** the FSDP paper (arXiv 2304.11277). PyTorch's FSDP tutorial.

### Lab 2 — Tensor Parallelism

**Required (~30 pages):** Megatron-LM §3, read twice: once for the algebra, once for the
f/g operators. Ultra-Scale Playbook tensor-parallelism chapter.

**Reference:** `megatron/core/tensor_parallel/layers.py` and `mappings.py`, *after* your
version passes. PyTorch's DTensor and `parallelize_module` tutorial.

**Optional:** Korthikanti et al. for the sequence-parallel stretch goal. Scaling Book's
training-parallelism chapter.

### Lab 3 — Pipeline Parallelism

**Required (~25 pages):** PTD-P §2.2. GPipe §2. Ultra-Scale Playbook pipeline chapter.

**Reference:** `torch.distributed.pipelining` docs. `picotron/pipeline_parallel.py`.

**Optional:** PipeDream (arXiv 1806.03377). PTD-P §2.2.2 for the interleaved schedule.

### Lab 4 — Capstone

**Required (~25 pages):** Scaling Book's parallelism-selection chapter. PTD-P §3 and its
empirical guidelines. Ultra-Scale Playbook's "finding the best configuration".

**Reference:** `pytorch/torchtitan` configs.

### Lab H — Hazards

**Required:** none. **Reference:** PyTorch distributed debugging docs, NCCL environment
variables, `py-spy` usage. All collected in [`DEBUGGING.md`](DEBUGGING.md).

---

## 4. The reading rhythm

Reading before implementing wastes time. Reading after failing is when it sticks. Per lab:

1. **Skim the Ultra-Scale Playbook section, 20 minutes.** Build a mental map, nothing more.
2. **Fill in `THEORY.md` and `predictions.yaml` by hand.** This is where the primary paper
   section gets read properly, because you need it to answer.
3. **Implement against the tests.**
4. **Then read picotron's or Megatron's version of the same component and diff it against
   yours.** This is the highest-value thirty minutes in each lab, and it only works in
   this order.
5. **Write the five "explain why" answers** in `runs/labN/writeup.md`, in your own words.
   An answer that is hard to write is the concept to re-read.

---

## 5. Structured courses

Free, optional, aligned.

- **Stanford CS336, Language Modeling from Scratch** (<https://cs336.stanford.edu>).
  Lectures on YouTube, assignments public. **Assignment 2, "Systems", is the closest
  institutional analogue to this repository**: profiling and benchmarking, FlashAttention-2
  in Triton, then distributed data parallel and optimizer state sharding. Assignment 1
  (BPE tokenizer, transformer, AdamW, all from PyTorch primitives) is excellent
  preparation if `common/reference_gpt.py` feels opaque. Their `assignment2-systems` repo
  is a good model for structuring starter code and tests, and it also uses `uv`.
- **UT Austin / Eijkhout HPC materials.** Companion lectures to *The Art of HPC*, for the
  MPI and collectives foundations.
- **PyTorch official distributed tutorials.** The API reference layer: the distributed
  overview, "Writing Distributed Applications with PyTorch", DDP, FSDP, Tensor Parallel
  with DeviceMesh, and `torch.distributed.pipelining`. Use as documentation, not as a
  curriculum.

---

## 6. Paid courses and certifications

Mentioned, not required.

- **NVIDIA Deep Learning Institute.** The relevant titles are *Scaling Deep Learning
  Training* and *Model Parallelism: Building and Deploying Large Neural Networks*. They
  include a cloud GPU environment and issue a certificate, so they answer the
  "corporate certification" question. Typically around $90 per self-paced course, and
  occasionally free during GTC. Verify the current catalogue at
  <https://nvidia.com/en-us/training/>; DLI's course list changes often.
- **AWS, GCP and Azure ML specialty certifications.** These certify *platform operation*,
  not distributed-training internals. Worth saying plainly so nobody buys one expecting
  this material.

**Position this honestly.** The strongest credential from this work is the repository
itself. A public repo containing a from-scratch DDP with measured overlap, a
tensor-parallel nanoGPT verified against a single-GPU reference, and a written scaling
analysis reads better to a hiring engineer than any certificate on the subject.

---

## 7. When a specific thing will not click

Short and clearly optional.

- **Lilian Weng, "How to Train Really Large Models on Many GPUs".** Best single-sitting
  overview of the landscape. Good to read *before* Lab 0 for orientation.
- **Sebastian Raschka, *Build a Large Language Model (From Scratch)*.** Appendix A covers
  PyTorch DDP gently. The rest is the best on-ramp if the transformer itself is shaky.
- **Karpathy's nanoGPT and the "Let's build GPT" video.** The origin of
  `common/reference_gpt.py`. Watch if the reference model is unfamiliar.
- **Kirk & Hwu, *Programming Massively Parallel Processors*, 4th ed.** The single-GPU layer
  below this curriculum: memory hierarchy, occupancy, coalescing. Relevant to *why* the
  arithmetic-intensity arguments hold. Not required.
- **Tanenbaum & van Steen, *Distributed Systems*, 4th ed.** (free PDF). Failure models,
  consistency, coordination. Broader than needed; useful context for Lab H.

---

## 8. Deliberately excluded

Stated with reasons, so the scope stays fixed.

- **FSDP and ZeRO-3 internals.** Mostly an engineering elaboration of ZeRO-1 plus
  parameter prefetching. High implementation cost, low marginal concept. Covered as
  reading plus an optional Tier-2 extension in Lab 1B.
- **Context and sequence parallelism, Ring Attention.** Genuinely important, and only
  earns its keep at sequence lengths you cannot reach here. Included as a Lab 2 stretch
  goal.
- **Expert parallelism and MoE.** A routing problem layered on top of these primitives.
  Better learned after.
- **Custom CUDA and Triton kernels.** An orthogonal skill: single-device performance, not
  distribution. CS336 Assignment 2 covers it well.
- **Multi-node networking, elastic training, fault tolerance.** Untestable without a real
  cluster, and a lab you cannot run is a lab you will not do.
