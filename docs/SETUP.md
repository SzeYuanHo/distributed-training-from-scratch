# Setup

From `git clone` to a running exercise, with repository-managed environments and caches
removable through `python tasks.py nuke`. Download and install time depends on the
selected Torch profile, platform and network.

Prerequisites: Git and Python 3.10 or newer.

---

## 1. The happy path

```bash
git clone <repo> && cd dist-train-from-scratch
```

Install the tested `uv` release if you do not have it. It is a separate user-level binary,
not part of the repository footprint:

```bash
curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh
```

```powershell
irm https://astral.sh/uv/0.11.16/install.ps1 | iex
```

Then:

```bash
python tasks.py setup      # venv, torch, deps, and the 1.1 MB dataset
python tasks.py doctor     # what this machine can and cannot run
python tasks.py test LAB=0 # your first wall of red. That wall is the to-do list.
```

`make setup` works if you have make. On Windows, `.\make.ps1 setup` or `make.cmd setup`.
All of them call `tasks.py`, so behaviour cannot drift between platforms.

**What `setup` does:** creates `./.venv` with `uv venv`, selects the requested Torch
backend, installs the locked dependency set, redirects the managed cache variables into
`./.cache` before installing, downloads tiny-shakespeare and checks its hash, then prints
the footprint and doctor verdict.

The manual CPU equivalent is:

```bash
uv venv .venv
uv pip sync --python .venv/bin/python --require-hashes \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu requirements/cpu.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
The explicit index strategy is needed because general packages come from PyPI while the
backend-specific Torch wheel comes from the PyTorch index. Every accepted artefact must
still match a hash recorded in the lock.

---

## 2. Why `uv` and not conda

|  | `uv` + `./.venv` | conda |
|---|---|---|
| Where the files go | inside the repo | `~/miniconda3/envs/<name>` |
| Removal | `rm -rf .venv` | `conda env remove` plus `conda clean -a`, and the package cache still lingers |
| Install time | network- and platform-dependent | network- and platform-dependent |
| Uninstall scope | managed venv and caches stay in the repo; the `uv` executable is separate | environments and package caches normally live outside the repo |

An in-repo venv keeps the large project environment and redirected caches visible
beside the source, which makes their ownership and removal straightforward. The
standalone `uv` executable and system-level software remain outside the repository.

Conda is supported. `env/environment.yml` has the recipe and a warning at the top saying
that `tasks.py nuke` cannot clean up after it, because it genuinely cannot.

---

## 3. Two install profiles

**`python tasks.py setup-cpu`** installs CPU-only torch. Plan for roughly 0.6 to
1.0 GiB with the current lock, varying by platform. It runs every Tier-0 and Tier-1
test in the repository: all correctness, collective-count assertions and schedule
visualisations. That is most of the curriculum.

**`python tasks.py setup-gpu`** installs CUDA torch and its platform-dependent runtime
wheels; plan for several gigabytes and measure the result locally. It adds single-GPU
profiling, memory measurement, and the ability to run Tier-2 code if you later get
compatible multi-GPU access.

Both install packages only into `./.venv`; the managed workflow also uses `./.cache`
for installer caches and `./data` for the verified corpus. Switching profiles is
`python tasks.py nuke-venv && python tasks.py setup-cpu`.

On Linux with a GPU, `setup` picks the GPU profile. On Windows it picks CPU even when a
GPU is present, for the reason in section 4.

---

## 4. Windows

Tier 0 works. Tier 1 works, with one caveat. Tier 2 does not exist here at all, because
NCCL is Linux-only.

**The caveat, and it is worth knowing before it costs you an afternoon.** Some CUDA
Windows wheels ship gloo with no usable transport. `torch.distributed.is_gloo_available()`
returns `True`, and then `init_process_group("gloo")` fails with:

```
RuntimeError: makeDeviceForHostname(): unsupported gloo device
```

Tier 1 runs on gloo, so that wheel cannot run the main development loop. The CPU-only
wheel from `download.pytorch.org/whl/cpu` works fine.

This is why `python tasks.py setup` picks the CPU profile on Windows even when it sees a
GPU: NCCL is unavailable there anyway, so the CUDA wheel would buy single-process
profiling and cost you the labs.

`python tasks.py doctor` **probes gloo directly** rather than trusting the availability
flag. It builds a one-rank group, runs an all-reduce, and reports what happened. If it
says gloo does not work, your options in order of effort are:

```bash
python tasks.py nuke-venv && python tasks.py setup-cpu
```

or WSL2, or the Docker image in section 6.

**You may also see a warning like this:**

```
[W socket.cpp:764] [c10d] The client socket has failed to connect to [MACHINE]:54257
(system error: 10049 - The requested address is not valid in its context.)
```

c10d may try the machine hostname before establishing the loopback connection it was
given. Treat this particular warning as informational only if the process group
subsequently initialises and the run proceeds normally. If startup fails, keep it as
diagnostic evidence. Exact wording and frequency vary by Torch build.

### WSL2

Recommended when you want a Linux userland on Windows. The same `torchrun`, gloo,
`mp.spawn` and NCCL APIs are available when the host and driver support them, and the
shell scripts work unmodified. A single WSL2 host still does not reproduce a real
cluster's network, topology or failure modes.

```powershell
wsl --install -d Ubuntu
```

Then clone the repo *inside* the WSL filesystem, not under `/mnt/c`. Cross-filesystem I/O
in WSL2 is slow enough to change your timing measurements.

**The WSL2 disk trap.** The `ext4.vhdx` virtual disk grows and never shrinks on its own.
Deleting files inside WSL2 does not give the space back to Windows. This is exactly the
forgotten-residue problem this repo is built to avoid, living one level below it.

To reclaim:

```powershell
wsl --shutdown
Optimize-VHD -Path "$env:LOCALAPPDATA\Packages\<distro>\LocalState\ext4.vhdx" -Mode Full
```

`Optimize-VHD` needs Hyper-V. Without it, use `diskpart`:

```
diskpart
select vdisk file="C:\Users\<you>\AppData\Local\Packages\<distro>\LocalState\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

---

## 5. macOS

Tiers 0 and 1 work. There is no Tier 2 without NVIDIA hardware, and MPS does not help:
`torch.distributed` has no MPS backend, so multi-process work runs on CPU.

```bash
python tasks.py setup-cpu
```

---

## 6. Docker

A nice-to-have, not the recommended path.

**Included:** a CPU-only image, roughly 1.5 to 2 GB built, that runs every Tier-0 and
Tier-1 test.

```bash
docker build -f docker/Dockerfile.cpu -t dtfs:cpu .
docker run --rm -it dtfs:cpu python scripts/test_answers.py -m "not tier2"
```

or

```bash
docker compose -f docker/docker-compose.yml run --rm tests
```

Worth it for a controlled CPU environment across machines, for a locked-down work laptop,
and because `docker rmi dtfs:cpu` is the cleanest uninstall of any option here. Host
kernel, CPU architecture and Docker runtime differences still apply.

**Deliberately excluded: a CUDA image.** Reasons, plainly:

- CUDA base images run 6 to 12 GB before your code, which is a real fraction of a 50 GB
  budget.
- GPU passthrough needs the NVIDIA Container Toolkit, and on Windows that means WSL2
  *plus* toolkit setup. Several failure points for zero conceptual gain.
- Docker Desktop on Windows keeps its own growing virtual disk with the same
  non-shrinking behaviour as WSL2.

If you want it anyway: start from `nvidia/cuda:12.6.0-runtime-ubuntu22.04`, install the
same packages, and run with `--gpus all`. Not required, not recommended under this disk
budget, and not tested here.

---

## 7. Kaggle: an optional two-GPU path

Kaggle has historically offered a two-T4 notebook accelerator suitable for Tier 2.
Availability, accelerator type, account eligibility, quota, session length, storage and
verification requirements can change. Check the current notebook settings and quota
display before planning around it.

When the **"GPU T4 x 2"** option is available, it provides two NVIDIA T4 devices with
16 GB each. The notebook checks the actual device count and NCCL availability at runtime
rather than assuming the advertised profile.

**Setup:**

1. Create an account and complete any verification currently required for accelerators.
2. Create a notebook. If offered, select **GPU T4 x 2** and enable Internet long enough
   to clone and install.
3. Open `notebooks/kaggle_2xT4_tier2.ipynb` or paste its cells in.
4. The setup cells clone the repo, install it with `--no-deps`, then install and validate
   every non-Torch project/test requirement explicitly. Prefer the runtime's existing
   CUDA-enabled torch build when it satisfies the supported range; replacing it can
   consume session time or install a wheel mismatched to the runtime.
5. Run `!nvidia-smi` and `!python scripts/doctor.py`. It must report 2 devices and NCCL
   available.
6. Labs 0, 1, 2 and 3 have Tier-2 tests; run the selected set with plain pytest:
   `!python -m pytest labs/labN -m tier2 -v --timeout=900`. Lab 1B and Lab 4
   have GPU benchmarks but no Tier-2 pytest cases. Run any benchmark driver with plain
   Python, for example
   `!python labs/labN/bench/<script>.py --nproc 2`.

**The mechanic that matters:** do not define the training entry point in a notebook cell
and then spawn it. Functions defined in a cell do not
pickle cleanly for `spawn`, and `fork` plus CUDA is unsafe. The repository's pytest
cases and benchmark drivers are already files and call `run_distributed` themselves,
so launch each exactly once with plain `pytest` or `python`. Wrapping either command in
`torchrun` creates a second, conflicting process topology. Use `torchrun` only for a
custom file whose top-level code is intentionally rank-aware, such as the notebook's
standalone NCCL probe.

**Two more things to know:**

- Hosted sessions are temporary and may stop for quota, duration or idleness; checkpoint
  anything that matters.
- **T4s are Turing generation and have no bf16 tensor cores.** Tier-2 code must use fp16
  or fp32, and must gate on `torch.cuda.is_bf16_supported()` rather than assuming.

---

## 8. Colab: one GPU, and that is the point

State this plainly because it is the most common misconception in this area:

**A Colab runtime exposing one GPU cannot run the multi-GPU labs.**

Hardware, availability, usage limits and idle policies vary by account and demand. Inspect
the assigned runtime rather than relying on a particular GPU model or duration.

Colab is still useful, and `notebooks/colab_single_gpu.ipynb` covers it: single-GPU
profiling, the memory-ledger measurements in Lab 1B, kernel timing, and any Tier-0 or
Tier-1 test when your laptop is unavailable.

---

## 9. Paid options

Optional. Provider availability, GPU types, quotas and prices change, so consult each
provider's current product and billing pages before relying on it.

- **Modal.** Supports explicit multi-GPU requests; `scripts/modal_tier2.py` is the
  optional entry point used here.
- **Lightning AI Studios, RunPod, Vast.ai and Lambda.** Possible hosted alternatives
  when their current offerings expose at least two compatible CUDA devices.
- **NVIDIA DLI.** Some courses include managed environments; see
  `docs/CURRICULUM.md`.

**Cost control.** Configure a provider budget or alert before the first run. Prefer
interruptible capacity when appropriate, and delete compute, volumes, snapshots and
other billable resources according to that provider's current billing model when done.

---

## 10. Troubleshooting

Every entry starts with `python tasks.py doctor`. It performs a bounded check of common
environment failures and reports which tiers are available; runtime varies, and
workload-specific failures can remain after it passes.

| Symptom | Cause | Fix |
|---|---|---|
| `makeDeviceForHostname(): unsupported gloo device` | CUDA Windows wheel with no gloo transport | Section 4. `nuke-venv` then `setup-cpu`, or WSL2. |
| Run hangs at startup, no output | NCCL with more ranks than GPUs, or ranks issuing different collectives | `python tasks.py kill`; use `python tasks.py test` for the CPU tiers, or expose enough GPUs before retrying Tier 2. See `docs/DEBUGGING.md`. |
| `Address already in use` | Stale `MASTER_PORT` from a crashed run | `python tasks.py kill`. `free_port()` is the default, so this is rare. |
| `Timed out initializing process group` | One rank crashed before `init_process_group` | Scroll up for the *first* traceback, not the timeout. |
| Tier 1 is slower than a single process | Expected | Not a bug. See the honesty rule in the README. |
| CPU pegged, everything crawls | Each rank spawning `OMP_NUM_THREADS = ncores` threads | Use `python tasks.py test` or `python tasks.py bench`; the repository environment sizes the thread pools for the selected process count. |
| CUDA OOM on an 8 GB card | Preset too large | Use `tiny`. Lab 1B teaches you to predict this instead of discovering it. |
| Notebook `spawn` fails to pickle | Entry point defined in a cell | Write it to a `.py` file. Run bundled self-spawning drivers with plain Python; reserve `torchrun` for custom rank-aware files. |
| Disk shrinking mysteriously on Windows | WSL2 or Docker Desktop vhdx growth | Section 4. |
| `ImportPathMismatchError` from pytest | Running pytest without this repo's config | Run from the repo root, or pass `--import-mode=importlib`. |

---

## 11. Removing everything

```bash
python tasks.py nuke              # deletes .venv, .cache, runs, data, __pycache__
python scripts/verify_clean.py    # reports watched global cache locations
```

`nuke` prints what it reclaimed. `verify_clean` checks a defined set of usual global cache
locations independently. To verify a run, take a `--snapshot` before the workload and
use `--check` afterward; CI runs that sequence around setup plus the reference suite.
It checks the repository-managed workflow's known fallback locations; it cannot account
for every operating system, package manager, filesystem path or third-party tool.
