# Hazard 2 — Postmortem

## 1. Symptom

Nothing looks wrong. The loss falls from about 1.15 to 0.07 over 60 steps, smoothly,
on every rank. The only thing that reveals it is the checksum the script prints at
the end: the four ranks' first-layer weight sums disagree in the second decimal
place.

## 2. First hypothesis

Written before investigating: I thought the gradient averaging had drifted, maybe a
missing divide or a rank skipping a step, and that the weights had slowly pulled
apart over the 60 steps. So I expected the divergence to grow with the step count.

Wrong, in a way worth recording. The divergence is not accumulated. It is there at
step 0 and the averaging never removes it.

## 3. How I confirmed it

`assert_close_across_ranks` on the parameters straight out of `build_model()`, before
a single forward pass. It failed immediately, which killed the accumulation theory
outright — no training had happened yet.

Then I checked the gradients after one backward: identical across ranks. The
all-reduce was fine all along.

## 4. Root cause

`build_model` calls `seed_everything(1337, rank_offset=True)`. Every rank draws its
weights from a different stream and nothing broadcasts afterwards, so four unrelated
models are being trained by one averaged gradient.

## 5. The fix

Seed with `rank_offset=False` for construction, then `sync_module_params` on the
finished module.

Why that one: seeding alike is an assumption about what consumes the RNG between the
seed call and the last layer's init. Add a dropout mask, a `torch.compile` decision,
or a layer whose init order changes, and the assumption quietly stops holding. The
broadcast is a guarantee and costs one collective at startup. Doing both means the
seed makes the broadcast a no-op in the common case, which is a nice property when
you are reading a diff.

I also moved the learning rate from 0.05 to 0.08. Once the ranks genuinely agree, 60
steps at 0.05 only reaches 0.59 of the starting loss and `test_fixed` wants half.
That is not part of the bug; it is the run being short.

## 6. The general lesson

The dangerous class is "correct machinery, wrong initial conditions". Every metric I
would normally look at was healthy, because averaging four wrong gradients is still
arithmetically a correct average. Only a cross-rank comparison sees it.

The habit: initialisation and sampling want opposite things from the RNG. Init wants
every rank identical, sampling wants every rank different. Two phases, seeded
explicitly at each, rather than one `seed_everything` at the top doing both jobs
badly.

## 7. How would you catch this earlier?

An assertion that parameters match across ranks, run once after construction and once
after any checkpoint load. Cheap, runs at startup, and turns this from a week-long
mystery into a startup failure. Checkpoint load matters as much as init: a resume
that only restores rank 0 reintroduces exactly this.

## 8. Time spent

Hypothesis to fix: about 25 minutes. Checking the weights before training rather than
after was what turned it around.
