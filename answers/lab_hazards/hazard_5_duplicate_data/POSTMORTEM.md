# Hazard 5 — Postmortem

## 1. Symptom

Four ranks converge faster than one process does, which is the wrong direction: four
ranks see four times the data per step, so the loss per step should fall at roughly
the same rate, not better. The script's own fingerprint table gives it away — all
four columns are identical at every step.

## 2. First hypothesis

Written before investigating: I assumed the seeding. `seed_everything(1337,
rank_offset=False)` sits right above the training loop, and my first thought was that
the loader was drawing its offsets from the global RNG, so an identical seed gave
every rank an identical draw. The fix I had in mind was `rank_offset=True`.

Wrong, and worth keeping. That fix would have produced four *different* batches and
still not the right ones — the union would not have reconstructed the global batch,
so the run would no longer be comparable with a single-process reference. It would
have moved the bug rather than removed it.

## 3. How I confirmed it

The fingerprints, which the script already prints. Identical across ranks at every
step, so the ranks are not merely correlated, they are reading the same rows.

That pointed at `get_my_batch` rather than at the RNG, because `common/data.py`
documents that offsets are drawn from `(seed, step)` alone and sliced by rank — there
is no per-rank randomness for the seeding to affect. Reading the call, `rank=0,
world_size=1` is passed literally.

The stronger check afterwards: concatenate the four ranks' batches at step 0 and
compare against a single-process batch of four times the size. Disjointness alone is
not enough, and that comparison is what the equivalence tests in Lab 1 quietly depend
on.

## 4. Root cause

`get_my_batch` hardcodes `rank=0, world_size=1`, so every rank asks for rank 0's slice
of a one-rank batch. Four copies of the same 4 sequences, and the all-reduce averages
four identical gradients into the gradient one rank would have computed alone.

## 5. The fix

Pass `dist_utils.rank()` and `dist_utils.world_size()`.

Why that one: the alternative is to shard outside the loader, slicing a global batch
in the training loop. That works and puts the sharding contract somewhere nobody
looks. `get_batch` already guarantees the union property by construction, and the
whole point of that guarantee is that the caller does not reimplement it.

## 6. The general lesson

This is four times the electricity for one rank's worth of learning, and every
number on the dashboard says it is going well. Faster convergence is not evidence of
anything — here it is the symptom, because the effective dataset is a quarter of what
it should be and the model is seeing the same examples repeatedly.

The habit: the data-sharding contract is a property to assert, not to infer from a
loss curve. Two lines and a hash.

## 7. How would you catch this earlier?

Fingerprint the batch per rank behind a debug flag, gather them at step 0, and assert
two things: that the fingerprints are all distinct, and that the concatenation
reconstructs the single-process batch. The second is the one that matters and the one
people skip.

Also worth knowing about the failure this loader sidesteps: sampling with replacement
means no rank ever runs out first. A loader that walks an epoch in order has to drop
or pad the ragged tail, or the rank that finishes early stops calling collectives and
the job hangs at the end of every epoch.

## 8. Time spent

Hypothesis to fix: about 10 minutes. The fingerprint table was already printed; I
just had to believe it over my theory about the seed.
