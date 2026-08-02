# answers/

This directory contains the completed reference work published with the repository.

The directories below mirror `labs/`. Each one can include the completed implementation,
filled-in theory and predictions, measurements, plots, reconciliation and writeup.
`labs/*/starter/` remains incomplete so a reader still has a clean exercise to attempt.

```
answers/
├── lab0_collectives/
├── lab1_ddp_bucketing/
├── lab1b_zero_memory/
├── lab2_tensor_parallel/
├── lab3_pipeline_parallel/
├── lab4_capstone_2d/
└── lab_hazards/
```

---

## How to use it

**If you are working through the labs:** write your code in `labs/*/starter/`, where the
tests look for it. Read the matching reference here after making your own attempt, or use
it earlier when your goal is to study a complete implementation rather than solve the
exercise unaided.

**If you are looking for a reference:** this is the runnable, completed counterpart to
the prose in each lab's `SOLUTION.md`. The files are published so readers can inspect the
whole implementation, reproduce the recorded work and compare approaches.

**If you are keeping your own finished work:** use `workspace/<lab>/`. That mirror is
gitignored so local paths, device information, traces and unfinished notes are not
published accidentally.

```
answers/lab1_ddp_bucketing/
├── starter/            completed implementation
├── THEORY.md           completed derivations
├── predictions.yaml    recorded pre-run predictions
└── runs/               measurements, plots, reconciliation and writeup
```

The `THEORY.md` and `predictions.yaml` are worth including. The gap between what someone
predicted and what they measured is more instructive than the code, and it is the part
nobody usually publishes.

The checked-in run artefacts are provenance from the original completed pass, not a
promise that another machine will reproduce the same timings. Where that pass exposed a
test- or benchmark-harness defect that was later repaired, the corresponding writeup
starts with a maintenance note. The original observation remains below it so readers can
see what prompted the repair; current CI is the authority on whether the reference code
passes now.

---

## Rules the repository enforces

**Exercise tests import `labs/*/starter/`, not this directory.** That keeps the learner's
red-to-green loop honest. CI separately materialises these completed implementations in a
temporary copy and runs the same test suite there, proving that the published reference
answers satisfy the contracts without overwriting the exercise stubs.

**Contents are tracked.** They are part of the published teaching material. Personal
work belongs in `workspace/`, whose empty directory structure is tracked with `.gitkeep`
files while its contents remain local by default.

---

## A note on the frozen reference

`common/reference_gpt.py` must not change casually, including in here. Every lab's correctness
claim is "my version matches the reference", and a solution that edited the reference to
make the comparison pass has not demonstrated anything.

`python tasks.py freeze-check` verifies this, and CI runs it. An intentional reference
change requires review followed by the explicit record command printed by the guard. If
you find yourself wanting to tweak the reference merely to make a test green, the
mismatch is telling you something about your sharding.
