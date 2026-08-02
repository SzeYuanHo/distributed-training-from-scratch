# Debugging distributed jobs

The actual day-to-day skill, and the one almost nobody teaches. `labs/lab_hazards` gives
you five broken jobs to practise on. This is the playbook.

---

## 0. Run `python tasks.py doctor` first

It is a bounded probe that checks the repository's common environment failure modes
before you spend time debugging the lab code. It probes gloo directly rather than
trusting a flag, reports device counts, and prints an explicit verdict about which tiers
this machine can run. A successful probe narrows the search; it does not prove every
driver, network, or workload-specific condition is healthy.

---

## 1. The three symptoms

Distributed bugs come in three shapes, and knowing which one you have decides everything
you do next.

**It hangs.** Silence, no error, no traceback. Almost always ranks disagreeing about
which collectives to call or in what order. Go to section 2.

**It crashes.** You have a traceback, which puts you ahead. The trick is finding the
*first* one. Go to section 3.

**It runs, and the numbers are wrong.** The dangerous one. Nothing errors, the loss falls,
the model is worse than it should be. Go to section 4.

---

## 2. Diagnosing a hang

### 2.1 Turn on the debug logging

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL python your_script.py
```

`DETAIL` logs every collective and checks that ranks agree about shapes and ordering. It
is slow and verbose and it is the first thing to reach for. `INFO` is a lighter version.

Read the *last* collective each rank logged. If rank 2's last line is an `allreduce` and
rank 3's is a `broadcast`, you have found it.

### 2.2 Dump the stacks

```bash
.venv/bin/python -m pip install py-spy
py-spy dump --pid 12345
```

`py-spy` reads a running process's python stack without stopping it. Do this for every
rank. On Windows use `.venv\Scripts\python.exe -m pip install py-spy`. This optional
debugger is not in the lock files; installing it enlarges the managed virtual environment.

**The counter-intuitive part, and the single most useful habit in this document:**

> Find the rank that is **not** blocked in a collective.

When rank 3 skips an all-reduce, ranks 0, 1 and 2 all block *inside* that all-reduce
waiting for a peer who will never arrive. Their stacks all point at the collective, which
makes the collective look like the problem. It is not. Rank 3 has moved on to something
else, and its stack shows you where the divergence happened.

So: dump every rank, discard the ones that agree with each other, and read the odd one
out.

### 2.3 Count the collectives

```python
from common import comm_counter, testing

with comm_counter.CommCounter() as c:
    train_step()

counts = testing.gather_object(c.counts())
if len(set(map(str, counts))) > 1:
    print(f"ranks disagree: {counts}")
```

If the counts differ, that difference *is* the hang. This is a cheap assertion to leave
in a training loop behind a debug flag.

### 2.4 The usual causes, in order of frequency

**A collective inside `if rank == 0:`.** By far the most common, and it almost always
arrives the same way: someone adds logging, the logging computes a global metric, and the
whole thing is guarded so it only prints once. Hazard 1.

**Uneven data across ranks.** One rank's dataloader runs out first, it stops calling
collectives, and everyone else waits at the end of the epoch. Drop the remainder or pad.

**A data-dependent branch.** `if loss > threshold:` where the loss differs across ranks.
Same shape as the first one, harder to spot.

**More ranks than GPUs with NCCL.** The normal launcher contract is one process with
exclusive access to each CUDA device. If two ranks are mapped onto one GPU, the first
collective can hang or fail instead of reporting oversubscription clearly.
`common.dist_utils.pick_backend` rejects that configuration and explains why.

**Symmetric blocking send/recv.** Every rank sends before any rank receives. Deadlocks
only when the message exceeds the transport's buffer, which is why it passes your tests
and fails in production. Hazard 4.

**Mismatched shapes in a collective.** Rank 0 all-reduces a `(128,)` tensor and rank 1 a
`(256,)` one. `TORCH_DISTRIBUTED_DEBUG=DETAIL` catches this specifically.

### 2.5 Make hangs fail instead

Three layers, and each catches something different:

```python
dist_utils.init(timeout_s=30)  # process group timeout
dist_utils.run_distributed(fn, join_timeout_s=60)  # outer join deadline
```

```python
@pytest.mark.timeout(120)                          # pytest-level
```

Lower the process-group timeout to 30 seconds while debugging. Waiting five minutes to
find out you deadlocked is five minutes you did not need to spend.

`torch.distributed.monitored_barrier()` raises naming the ranks that failed to arrive,
instead of hanging. Too expensive for a training loop, excellent for bisecting one.

### 2.6 At scale

**Flight Recorder.** Keeps a ring buffer of recent collectives per rank and dumps it on
timeout:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
```

Built for exactly this problem at thousand-GPU scale, where `py-spy`-ing every rank is not
an option. torch 2.3 and later.

---

## 3. Diagnosing a crash

**Find the first traceback, not the loudest one.** When rank 3 crashes, the other ranks
hit their collective timeout thirty seconds later and produce their own, longer,
scarier-looking errors. Those are consequences. Scroll up.

**Write per-rank log files.** Above two ranks, interleaved stdout is unreadable:

```python
import sys

sys.stdout = open(f"runs/rank_{dist_utils.rank()}.log", "w", buffering=1)
```

Then `grep -n Error runs/rank_*.log` and look at the earliest timestamp.

**`common.dist_utils.printr`** gives you rank-prefixed, barrier-ordered output for small
runs. It costs one barrier per rank per call, so keep it out of hot loops.

---

## 4. The silent wrong answer

The dangerous category. Nothing raises. The loss still falls. Only comparison against a
reference catches these.

### The checklist

Work through it in order. Each item has a test in this repo that demonstrates it.

- [ ] **Are the model weights identical across ranks?**
      `testing.assert_close_across_ranks(p.data)` on every parameter after construction.
      Different weights means you are training N unrelated models and averaging their
      gradients. Hazard 2.

- [ ] **Are gradients divided by the world size?**
      `all_reduce` with `SUM` gives N times the mean. Skipping the divide makes the
      effective learning rate N times too large. The signature is that every element is
      off by the same integer factor, and
      `common.testing.compare_to_reference` spots that and says so.

- [ ] **Was every async handle waited on before `.grad` was read?**
      Between `async_op=True` and `wait()` the tensor is undefined, not "probably
      finished". Hazard 3.

- [ ] **Does each rank see different data?**
      Hash it: `data.batch_fingerprint(x)`, gather, assert the hashes differ. Every rank
      reading the same shard makes the effective dataset 1/N the size. Hazard 5.

- [ ] **In tensor parallelism, are `f` and `g` the right way round?**
      A swap leaves the forward output correct and the gradients wrong. Only a gradient
      comparison catches it. Lab 2's `test_negative.py`.

- [ ] **Is a row-parallel bias added once, after the all-reduce?**
      Added before, it appears N times in the sum.

- [ ] **Are in-model collectives inside a `torch.autograd.Function`?**
      A bare `dist.all_reduce` on a tracked tensor detaches the graph silently.

- [ ] **In a pipeline, did the received activation get `requires_grad_(True)`?**
      Without it the backward stops at each stage boundary and the early layers never
      train.

### The habit

A falling loss is never evidence of correctness. Every item above produces one.

Compare against a frozen single-process reference. That is why every lab in this repo has
one, and why "the loss went down" is not accepted as evidence anywhere in it.

---

## 5. Environment variables worth knowing

| Variable | What it does |
|---|---|
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Log every collective, validate shapes and ordering. First thing for a hang. |
| `TORCH_CPP_LOG_LEVEL=INFO` | c10d's own logging, below the python layer. |
| `NCCL_DEBUG=INFO` | NCCL's setup and topology decisions. Tier 2 only. |
| `NCCL_DEBUG_SUBSYS=TUNING` | Show NCCL's algorithm and protocol selection. `COLL` instead logs collective operations. |
| `NCCL_ALGO=Ring` / `Tree` | Force two available algorithms for a controlled comparison; normal selection may include other algorithms and protocols. |
| `TORCH_NCCL_TRACE_BUFFER_SIZE=1000` | Flight Recorder ring buffer. |
| `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | Dump that buffer when a collective times out. |
| `CUDA_LAUNCH_BLOCKING=1` | Make CUDA errors point at the right line. Slow; debugging only. |
| `OMP_NUM_THREADS=N` | Threads per rank. The task runner and repository `conftest.py` set this; custom entry points must do so before importing torch. |
| `TORCH_SHOW_CPP_STACKTRACES=1` | C++ frames in tracebacks, for crashes inside c10d. |

---

## 6. Cleaning up after a bad run

```bash
python tasks.py kill      # reap orphaned ranks holding ports and cores
python tasks.py clean     # drop runs/ and caches, keep .venv and data
```

A killed distributed launch—whether it used the repository's spawn helper or a custom
`torchrun` command—can leave children alive. They hold the master port, so your next run
fails with "Address already in use", and they hold a core each, so your laptop fan stays
on. `tasks.py kill` finds them by command line, shows you the list, and asks before
terminating.

If ports are still held after that, wait a few seconds for TIME_WAIT to clear.

---

## 7. Reading the noisy warnings

Some output looks alarming and is not.

**`[W socket.cpp:764] [c10d] The client socket has failed to connect to [MACHINE]:54257`**
can appear on Windows when c10d tries the machine hostname before a usable loopback
connection. Treat it as informational only when the process group subsequently
initialises and the run continues; otherwise it is part of the failure, not a warning to
ignore. The exact wording and frequency vary with the Torch build and launch topology.

**`Warning: Grad strides do not match bucket view strides`** from DDP. A performance note,
not a correctness one: a gradient is not contiguous, so DDP has to copy it.

**`UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`** after a
skipped step, usually from a gradient-scaler overflow. Expected under fp16.

**Nothing at all** is the one to worry about. See section 2.
