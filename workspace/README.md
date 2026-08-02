# workspace/

This is the learner-owned side of the repository.

The three relevant locations have deliberately different jobs:

- `labs/<lab>/starter/` contains the incomplete exercise code that the tests import.
- `answers/<lab>/` contains the published, completed reference implementation,
  derivations, measurements and writeups.
- `workspace/<lab>/` is an optional private copy area for your own implementation,
  notes, measurements and machine-specific details.

Start by working in `labs/<lab>/starter/`, because that is where the tests look. If you
want to keep a separate record, copy the files you changed plus your filled-in theory,
predictions and run artefacts into the matching directory here.

Everything below this README is ignored except `.gitkeep` files. A normal `git add .`
therefore does not publish local GPU names, paths, profiler traces, unfinished work or
other device-specific details. Remove or override that ignore rule only when you
deliberately want to publish your own work.

The empty mirror is:

```text
workspace/
├── lab0_collectives/
├── lab1_ddp_bucketing/
├── lab1b_zero_memory/
├── lab2_tensor_parallel/
├── lab3_pipeline_parallel/
├── lab4_capstone_2d/
└── lab_hazards/
```
