# Lab H — Hazards: five broken jobs to diagnose

Debugging distributed training is the day-to-day job and it is almost never taught. Five
scripts here are broken in five different ways. Your task is to find out why, fix them,
and write up what you learned.

Do these after Lab 1. Hazards 1, 2, 3 and 5 need nothing beyond Lab 0. Hazard 4 makes more
sense after Lab 3, but you can attempt it earlier.

---

## The rules

For each hazard:

1. **Run `broken.py`.** Observe the symptom. Do not read the code yet.
2. **Form a hypothesis.** Write it in `POSTMORTEM.md` before you investigate. Being wrong
   in writing is the useful part.
3. **Confirm it with a tool**, not by reading the source. Reading the source is how you
   find the bug you already suspected; the tools are how you find the one you did not.
4. **Fix `broken.py` in place**, then make `test_fixed.py` pass.
5. **Finish the postmortem.** Ten minutes. The general lesson section is the one that
   pays back.

The whole set takes about half a day. Skipping the postmortems saves you two hours and
most of the value.

---

## The five

| # | Directory | Symptom |
|---|---|---|
| 1 | `hazard_1_the_hang` | Silence. The job starts and never finishes, no output, no error. |
| 2 | `hazard_2_the_desync` | The loss falls nicely. The saved model is garbage. |
| 3 | `hazard_3_silent_gradient` | The loss falls slightly more slowly than it should. Nothing errors. |
| 4 | `hazard_4_pipeline_deadlock` | The stages exchange activations and the job never finishes. |
| 5 | `hazard_5_duplicate_data` | Converges suspiciously fast, then the validation loss climbs. |

Every one represents a real class of distributed-training failure.

---

## The toolkit

`docs/DEBUGGING.md` has the full playbook. The short version:

**Start with `python tasks.py doctor`.** It is a bounded check of the common environment
failure modes and reports which tiers this machine can run. Runtime varies by machine,
and a successful probe narrows the search rather than proving the whole environment.

**`TORCH_DISTRIBUTED_DEBUG=DETAIL`** logs every collective and checks that ranks agree on
shapes and ordering. The first thing to turn on for a hang.

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL python labs/lab_hazards/hazard_1_the_hang/broken.py
```

**`py-spy dump --pid <pid>`** shows the python stack of a running process without
stopping it. On a hang, dump every rank and find the one that is *not* sitting in a
collective. That rank is the culprit; the others are its victims.

```bash
.venv/bin/python -m pip install py-spy
py-spy dump --pid 12345
```

On Windows use `.venv\Scripts\python.exe -m pip install py-spy`. This optional tool is
not in the lock files, so installing it adds to the managed virtual environment.

**`common.testing.assert_close_across_ranks`** proves whether a tensor that should be
replicated actually is. The direct detector for hazard 2.

**`common.comm_counter.CommCounter`** shows what was sent and how much. When ranks
disagree about the collective count, you have found your hang.

**`common.data.batch_fingerprint`** hashes a batch so you can compare across ranks
without moving the data. Hazard 5.

**Timeouts.** Every hazard script sets a short process-group timeout, so a hang fails in
30 seconds rather than sitting there. In your own code, do the same: `init(timeout_s=30)`
while debugging.

---

## Reading a hang

The single most useful habit, and it is counter-intuitive:

**Find the rank that is *not* blocked in a collective.**

When rank 3 skips an all-reduce, ranks 0, 1 and 2 block inside it waiting for a peer that
will never arrive. Their stacks all point at the collective, which looks like the
collective is the problem. It is not. Rank 3 has moved on to something else, and its stack
is the one that tells you where the divergence happened.

So: dump every rank, discard the ones that agree with each other, and look at the odd one
out.

---

## Where this lives in production

- **`TORCH_DISTRIBUTED_DEBUG`** is implemented in
  `torch/csrc/distributed/c10d/Utils.hpp` and the logging it enables lives in
  `ProcessGroupWrapper`.
- **Flight Recorder** keeps a ring buffer of recent collectives per rank when
  `TORCH_NCCL_TRACE_BUFFER_SIZE=1000` is set. Add
  `TORCH_NCCL_DUMP_ON_TIMEOUT=1` to write that buffer when the watchdog times out.
- **`NCCL_DEBUG=INFO`** with `NCCL_DEBUG_SUBSYS=TUNING` reports algorithm and protocol
  selection. `COLL` logs collective operations instead. Tier 2 only.
- **`torch.distributed.monitored_barrier`** raises with the ranks that failed to arrive
  instead of hanging. Worth knowing about; expensive enough that you would not leave it
  in a training loop.
