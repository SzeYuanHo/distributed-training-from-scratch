# Footprint

Repository-managed environments, caches, data and generated artefacts live inside this
directory. Deleting the folder reclaims those items; system tools, drivers, container
storage and software launched outside the managed commands are separate.

That is a claim, and claims should be checked, so:

```bash
python tasks.py disk            # what it costs right now
python tasks.py disk WRITE=1    # rewrite the measured column below
python tasks.py nuke            # delete it all, print what came back
python scripts/verify_clean.py  # report the watched global cache locations
```

To verify a managed workload, run `python scripts/verify_clean.py --snapshot`, run the
workload, then run `python scripts/verify_clean.py --check`. CI performs that sequence
on every push: it snapshots a defined set of usual global cache locations before setup,
runs the completed-answer suite, then compares the locations. A new file in a watched
path or growth beyond the configured tolerance fails the build. This is a scoped cache
check, not a monitor of every filesystem location or system tool.

---

## Budget and measurement

<!-- measured: intentionally local; run `python tasks.py disk WRITE=1` -->

| Item | Budget | Measured | Notes |
|---|---|---|---|
| source + docs | < 10 MiB | measure locally | tracked source, published answers and docs |
| `.venv`, GPU path | plan for 4–6 GiB | measure locally | CUDA torch plus platform-dependent runtime wheels; verify on the target platform |
| `.venv`, CPU-only path | plan for 0.6–1.0 GiB | measure locally | enough for Tiers 0 and 1, which is most of the curriculum |
| `data/` | ~6 MiB | measure locally | 1.12 MB of text plus a 2.2 MiB uint16 token array |
| `data/` with TinyStories | ≤ 500 MiB | n/a by default | hard-capped, behind `DTFS_ALLOW_TINYSTORIES=1` |
| `runs/` checkpoints | ≤ 200 MiB | measure locally | budget only; remove old checkpoints with `tasks.py clean` |
| `runs/` profiler traces | ≤ 500 MiB | measure locally | `timers.trace` warns at its threshold; it does not truncate or auto-prune |
| `.cache/` | ≤ 1 GiB | measure locally | mostly the uv download cache. Purgeable any time with `tasks.py clean`. |
| **Total, GPU path** | **plan for ≤ 8 GiB** | measure locally | conservative allowance; the locked wheel set is platform-dependent |
| **Total, CPU-only path** | **plan for ≤ 2 GiB** | measure locally | includes the environment, data and cache allowance |

Portable measurements are not committed because they depend on platform, Python, the
selected Torch backend and local artefacts. Run `python tasks.py disk WRITE=1` to replace
the `measure locally` entries in your own checkout.

---

## What goes where

```
.venv/     the repository-managed virtualenv
.cache/    managed third-party cache variables, set before torch loads:
             huggingface/  torch/  triton/  inductor/  pip/  uv/
             matplotlib/   wandb/
data/      tiny-shakespeare and its uint16 token array
runs/      plots, writeups, measurements, traces, checkpoints
traces/    profiler traces written separately from other run artefacts
```

All five are gitignored. All five are removed by `python tasks.py nuke`.

---

## How the redirection works

`common/repo_env.py` sets fourteen environment variables before anything imports torch:

`XDG_CACHE_HOME`, `HF_HOME`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `MPLCONFIGDIR`, `WANDB_DIR`,
`WANDB_MODE=offline`, `TOKENIZERS_PARALLELISM=false`, plus `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` sized to the rank count.

`conftest.py` and `tasks.py` both call `repo_env.apply()` as their first action, before any
torch import. `activate.sh` and `activate.ps1` set the same variables for interactive
shells.

The ordering matters. Torch decides where its cache lives at import time, so a variable
set afterwards is ignored, and you would get a repo that looks clean and quietly seeds
`~/.cache/torch` anyway.

`WANDB_MODE=offline` is deliberate: a teaching repository has no business uploading
anything, and an accidental `wandb.init()` should write to `.cache/wandb` rather than to
someone's account.

---

## What this does not cover

Three things live one level below the repo and it cannot clean them up.

**`uv` itself** lives in the location chosen by its installer or package manager.
If you want to remove it, first resolve the exact executable with `which uv` or
`Get-Command uv`, then use the matching installer's documented uninstall procedure.

**conda**, if you took that route. `env/environment.yml` warns about this at the top:
a normal user-level configuration keeps environments and its package cache outside the
repository. Check `conda info` for the actual paths, and remove this environment with
`conda env remove -n dist-train`; `conda clean -a` is the broader optional cache cleanup.

**WSL2 and Docker Desktop virtual disks.** `ext4.vhdx` grows and never shrinks on its own,
so deleting files inside WSL2 does not return the space to Windows. This is exactly the
forgotten-residue problem, one layer down. The reclaim procedure is in
[SETUP.md](SETUP.md) section 4.

---

## A note on the venv size

The GPU path is several gibibytes and most of it is the CUDA runtime shipped inside
platform-specific dependency wheels: cuBLAS, cuDNN, cuSPARSE, NCCL and friends. The
exact set and size depend on platform and release; `python tasks.py disk` is the
authority for the current checkout.

Which is why the CPU-only profile is the recommended default here. Plan for roughly
0.6 to 1.0 GiB with the current locked environment; it runs every Tier-0 and Tier-1
test in the repository, meaning all correctness, collective-count assertions and
schedule visualisations. The CUDA profile adds single-GPU profiling, device-memory
measurement and the runtime needed for Tier 2 when multiple compatible GPUs are
available.

If disk is tight: `python tasks.py setup-cpu`, and add the GPU profile later with
`python tasks.py nuke-venv && python tasks.py setup-gpu`.
