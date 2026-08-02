# dist-train-from-scratch

Build distributed training from the primitives up: collectives, gradient bucketing,
ZeRO sharding, tensor parallelism, pipeline schedules. Seven labs, each finished when
its tests go green.

You need one machine. Most of the curriculum runs on a laptop CPU in seconds per test.

---

## Quickstart

Prerequisites: Git and Python 3.10 or newer.

```bash
git clone <repo> && cd dist-train-from-scratch
python tasks.py setup          # everything lands in ./.venv and ./.cache
python tasks.py doctor         # tells you which tiers this machine can run
python tasks.py test LAB=0     # first red wall; that wall is your to-do list
```

`make setup` works too if you have make. On Windows, `.\make.ps1 setup`. All three
call the same code in [tasks.py](tasks.py).

This is a clone-first curriculum, not a standalone library distribution. The
editable package exposes `common/` so the labs can import shared utilities; the
lab manuals, exercises, answers and task runner are repository content and are
validated directly by CI rather than bundled into the wheel.

---

## The one hard constraint, stated up front

Distributed training needs many GPUs. You probably have one. In the normal NCCL
process-per-GPU model, every rank needs exclusive access to its assigned CUDA device.
Launching two ranks when only one GPU is available gives both processes the same device;
collectives can then hang or fail instead of producing a useful oversubscription error.

The repo is built around that. Here is what actually needs multiple GPUs:

| What you want to learn | Needs multiple GPUs? |
|---|---|
| Collective algorithms and their cost models | no |
| Sharding math: does my TP layer compute the right thing? | no |
| Autograd interaction with collectives | no |
| Bucketing, scheduling, deadlock avoidance, rank desync | no |
| Memory accounting and parallelism trade-offs | no |
| Wall-clock speedup and real overlap timelines | **yes** |

Around 85% of the material needs multiple *processes*, not multiple *devices*.
Several CPU processes on the gloo backend give you real `torch.distributed`, real
collectives, real autograd hooks, real deadlocks, and real correctness bugs, in
seconds per run.

That is not a workaround invented here. Hugging Face's teaching framework
[picotron](https://github.com/huggingface/picotron) ships a CPU mode and documents
2-way data by 2-way tensor by 2-way pipeline parallelism across 8 CPU processes on
one machine. Same technique, and this repo is explicit about where it stops being
enough.

---

## Three tiers

The tier is a flag, not a fork of the code.

**Tier 0 — math.** One process, no `torch.distributed` at all. Shard the weights by
hand, do the arithmetic, assert the sharded result equals the reference. Runs in
under a second. This is where the linear algebra gets learned, before any
communication exists to confuse the picture.

**Tier 1 — semantics.** N CPU processes on gloo, on your laptop. Real ranks, real
collectives, real autograd, real async ops. Verify numerical equivalence against a
single-process reference and count how many collectives fired and how many bytes
moved. Seconds to a few minutes. **This is the default development loop and where
you will spend most of your time.**

**Tier 2 — performance.** Two CUDA-capable GPUs with NCCL. The only tier that can
honestly show comm/compute overlap, NCCL algorithm selection, and wall-clock scaling.
Kaggle has historically offered a two-T4 accelerator, but accelerator types, eligibility,
session limits and weekly quotas are account- and demand-dependent. Check the current
Kaggle settings before relying on it. A single-GPU Colab session cannot run the
multi-GPU labs.

Your local GPU, if you have one, handles single-process work: profiling, kernel
timing, memory measurement, and the single-GPU baselines the Tier-1 tests compare
against.

> **Honesty rule.** Tier 1 teaches *what happens*. Tier 2 teaches *how fast*. A
> gloo/CPU timing is never evidence of a distributed speedup. Your from-scratch DDP
> will measure **slower** than single-process at Tier 1. That is not a bug you
> introduced. On one machine, communication dominates, and every lab README states
> the expected Tier-1 timings so it is never a surprise.

---

## The labs

| Lab | Title | Effort | The question it answers |
|---|---|---|---|
| [0](labs/lab0_collectives) | Collectives & Cost Models | 0.5–1 day | What is an all-reduce, and what does it cost? |
| [1](labs/lab1_ddp_bucketing) | DDP from Scratch: Bucketing & Overlap | 1.5–2 days | How do you hide communication behind computation? |
| [1B](labs/lab1b_zero_memory) | ZeRO-1 & the Memory Ledger | 0.5 day | Where does the memory actually go? |
| [2](labs/lab2_tensor_parallel) | Megatron-Style Tensor Parallelism | 2–3 days | How do you split a matmul across devices and keep autograd correct? |
| [3](labs/lab3_pipeline_parallel) | Pipeline Parallelism: GPipe to 1F1B | 1–2 days | How do you split by depth, and what does the bubble cost? |
| [4](labs/lab4_capstone_2d) | Capstone: 2D Composition & Scaling Report | 1 day | Given a model and a cluster, which parallelism do you choose? |
| [H](labs/lab_hazards) | Hazards: five broken jobs to diagnose | 0.5 day | How do you debug a hung 8-rank job? |

Roughly 6 to 9 days total. **0 → 1 → 2 is the required spine.** Lab 3 is strongly
recommended and cheap to implement on CPU. Lab 4 is where the pieces become a
decision-making skill. Labs 1, 2, and 3 depend only on Lab 0 and `common/`, so you
can reorder or drop them without breaking anything.

Time-boxed to three days: 0 → 1 → 2, skipping stretch goals.

---

## How a lab works

Each numbered implementation lab holds the same eight things:

```
labs/labN_topic/
├── README.md          the lab manual, with a consistent section structure
├── THEORY.md          derivations to work by hand, with blanks
├── predictions.yaml   you fill this in BEFORE running anything
├── starter/           skeletons raising NotImplementedError
├── reference/         single-process ground truth the tests compare against
├── tests/             pytest. This is the definition of done.
├── bench/             instrumentation producing the lab's deliverable plot
└── SOLUTION.md        prose walkthrough, deliberately not copy-pasteable
```

`lab_hazards/` is the deliberate exception: it is a set of small broken
programs, hints and postmortems rather than another implementation scaffold.

The incomplete exercise and completed reference are intentionally separate:

- work in `labs/*/starter/`; the exercise tests import those files;
- read the completed implementation and recorded results in [`answers/`](answers/README.md);
- keep private copies, local measurements and device-specific notes in
  [`workspace/`](workspace/README.md), which is ignored except for its empty mirror.

Four principles shape all of them:

**Verification before performance.** A lab is done when `pytest` passes, not when a
loss curve looks nice. Every lab has a frozen single-process reference it must match
numerically. That makes "am I finished?" objective.

**A falling loss is never evidence of correctness.** Every silent bug in this field
produces a falling loss: unbroadcast weights, gradients summed but not averaged, the
tensor-parallel `f` and `g` operators swapped, every rank reading the same data
shard. Only comparison against the reference catches them, which is why every lab
has one.

**Predict, then measure, then reconcile.** Before running anything you write your
numeric prediction into `predictions.yaml`: bytes on the wire, peak memory, bubble
fraction, step time. The harness measures the real value and prints both. Being
wrong is the learning event, and it is what turns "I implemented it" into "I
understand it".

**Failure is curriculum.** Deadlocks, rank desync, silently wrong gradients, and
NCCL timeouts get their own lab. Debugging distributed jobs is the actual job.

---

## Commands

```bash
python tasks.py setup           # venv + deps, selects the documented platform profile
python tasks.py setup-cpu       # force the smaller CPU-only environment
python tasks.py doctor          # environment self-check and tier verdict
python tasks.py test            # all Tier-0 and Tier-1 tests
python tasks.py test-answers    # validate published answers in an isolated copy
python tasks.py test LAB=2      # one lab
python tasks.py test-tier2 LAB=2 # self-spawning Tier-2 tests; needs 2+ GPUs
python tasks.py bench LAB=1     # produce the lab's deliverable plots
python tasks.py reconcile LAB=1 # predictions vs measurements
python tasks.py gantt LAB=3     # render the schedule visualisation
python tasks.py kill            # reap orphaned ranks after a hang
python tasks.py disk            # per-directory footprint
python tasks.py clean           # drop runs/ and caches, keep .venv and data
python tasks.py nuke            # remove everything generated, print bytes reclaimed
```

---

## The footprint promise

Repository-managed commands redirect the caches they control into this directory.

The managed cache variables point at `./.cache` before torch loads: Hugging Face,
torch hub, Triton, Inductor, pip, uv, matplotlib and wandb. The virtualenv lives at
`./.venv`. Deleting the folder removes the repository-managed environment and caches.
Tools launched outside these commands, system package managers, GPU drivers and the
standalone `uv` executable remain outside the repository.

`python tasks.py nuke` does it for you and prints what came back.
`python scripts/verify_clean.py` checks the managed claim independently by snapshotting
the usual global cache locations before and after setup plus the completed-answer suite,
and CI runs that check on every push. This verifies the locations the checker knows
about; it is not a claim that an operating system or third-party tool can never write
elsewhere.

| Item | Budget |
|---|---|
| Source and docs | < 10 MiB |
| `.venv`, GPU path | plan for roughly 4–6 GiB; wheel sets vary by platform |
| `.venv`, CPU-only path | plan for roughly 0.6–1.0 GiB |
| `data/` | ~3.2 MiB after tokenisation (tiny-shakespeare) |
| `runs/` | ≤ 700 MiB budget, checked by `tasks.py disk`; clean manually |
| `.cache/` | ≤ 1 GiB, purgeable any time |

Measured numbers for your machine: `python tasks.py disk`. See
[docs/FOOTPRINT.md](docs/FOOTPRINT.md).

---

## Platform notes

**Linux and WSL2** are the smoothest path. All curriculum tiers are available,
including gloo and NCCL when the required GPUs are present.

**Native Windows** runs Tier 0 fine. Tier 1 depends on whether your PyTorch wheel
shipped a usable gloo transport, and several Windows builds have not.
`python tasks.py doctor` probes this directly rather than trusting
`is_gloo_available()`, and tells you which of WSL2, Docker, or Kaggle to use if gloo
is unusable. See [docs/SETUP.md](docs/SETUP.md) for the details and the WSL2 disk
reclaim procedure, which matters because `ext4.vhdx` grows and never shrinks on its
own.

**macOS** runs Tiers 0 and 1. There is no Tier 2 without NVIDIA hardware.

---

## Reading

The repo is a lab manual for a small set of texts, not a replacement for them. Three
anchors, and a long bibliography is a way of avoiding reading anything:

- **[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)**
  (Hugging Face, 2025). Free, ~150 pages, and the closest match to this curriculum
  that exists. The main course text.
- **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google
  DeepMind, 2025). Free. Where you learn to predict rather than measure. Chapter 1
  on rooflines is required for Lab 0.
- **Introduction to Parallel Computing**, Grama, Gupta, Karypis & Kumar, 2nd ed.
  Chapter 4 is Lab 0 in book form. Nothing in modern distributed training is new
  here; it is the same collectives and the same cost models applied to a new
  workload.

Per-lab reading maps, papers by section, and the free alternatives are in
[docs/CURRICULUM.md](docs/CURRICULUM.md). Full citations in
[references.bib](references.bib).

---

## Not goals

Stated so the scope stays fixed.

Not a training framework. Not performance-competitive. Not multi-node. Not a
substitute for the primary texts. No lab requires training to convergence, and
nothing here aims to produce a good model.

Deliberately excluded, with reasons in
[the curriculum](docs/CURRICULUM.md#8-deliberately-excluded):
FSDP/ZeRO-3 internals (an engineering elaboration of ZeRO-1, covered as reading plus
a Tier-2 demo), context and sequence parallelism (only earns its keep at sequence
lengths you cannot reach here, included as a Lab 2 stretch goal), expert parallelism
(a routing problem layered on these primitives, better learned after), custom
CUDA/Triton kernels (single-device performance, an orthogonal skill), and multi-node
networking, elastic training, and fault tolerance (untestable without a real
cluster).

---

## What finishing looks like

Without reference material, you should be able to:

1. Derive the bytes on the wire and the latency of a ring all-reduce for N ranks and
   M bytes, and explain when a latency-optimal algorithm beats a bandwidth-optimal
   one.
2. Explain why DDP buckets gradients in reverse parameter-registration order, and
   derive the bucket-size trade-off.
3. Write the memory ledger for mixed-precision AdamW training and show what each
   ZeRO stage removes.
4. State where the two conjugate communication operators sit in a tensor-parallel
   block, and explain why forward and backward need *different* collectives.
5. Derive the pipeline bubble fraction and explain why 1F1B has the same bubble as
   GPipe but much lower activation memory.
6. Given a model size, a GPU count, and an interconnect, argue for a parallelism
   configuration with numbers rather than vibes.

The repository itself is the credential. A public repo containing a from-scratch DDP
with measured overlap, a tensor-parallel nanoGPT verified against a single-GPU
reference, and a written scaling analysis reads better to a hiring engineer than any
certificate on the subject.

---

## Solutions

The repository publishes both an exercise and a completed reference without putting the
answer beside the file being edited:

- `tests/`, the objective contract, always present.
- `SOLUTION.md` per lab: prose and small annotated fragments explaining the approach
  to each step.
- [`answers/`](answers/README.md), the completed implementations, filled derivations,
  predictions, measurements, plots and writeups used as the published reference.
- [`workspace/`](workspace/README.md), an empty, gitignored mirror for a reader's own
  copies and machine-specific output. `labs/*/starter/` stays pristine in the published
  branch so the next person can still start from the exercise.

CI copies `answers/` over `labs/*/starter/` only inside a temporary checkout and runs the
same suite there. The green build therefore certifies the published answers while the
checked-in exercise remains incomplete.

---

## Attribution

`common/reference_gpt.py` is adapted from
[nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT). The
tiny-shakespeare corpus is downloaded at setup time from Karpathy's char-rnn
repository and is not redistributed here. This repository is MIT licensed; see
[LICENSE](LICENSE).
