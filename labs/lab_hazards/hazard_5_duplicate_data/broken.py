#!/usr/bin/env python3
"""HAZARD 5 — The duplicate data.

Symptom
-------
Training converges noticeably faster than the single-process baseline. You
congratulate yourself on the speedup from data parallelism.

Then the validation loss starts climbing while the training loss keeps falling.

::

    python labs/lab_hazards/hazard_5_duplicate_data/broken.py

Your job
--------
Work out why four ranks are learning less than four ranks' worth. Fix it and make
``test_fixed.py`` pass.

The symptom points at overfitting, which is a modelling problem. It is not a
modelling problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common import repo_env  # noqa: E402

repo_env.apply()

import torch  # noqa: E402

from common import collectives, data as data_mod, dist_utils, reference_gpt  # noqa: E402

STEPS = 30
BATCH_PER_RANK = 4
BLOCK = 32


def get_my_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's slice of the global batch for this step.

    Shards the data across ranks so that each rank sees a different part of the
    corpus and the union covers the whole global batch.
    """
    dataset = data_mod.load_dataset()

    return data_mod.get_batch(
        "train",
        batch_size=BATCH_PER_RANK,
        block_size=BLOCK,
        rank=0,
        world_size=1,
        step=step,
        dataset=dataset,
    )


def worker() -> dict:
    rank, world = dist_utils.rank(), dist_utils.world_size()
    dataset = data_mod.load_dataset()

    dist_utils.seed_everything(1337, rank_offset=False)
    model = reference_gpt.make_model("tiny", block_size=BLOCK, vocab_size=dataset.vocab_size)
    dist_utils.sync_module_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    losses = []
    fingerprints = []

    for step in range(STEPS):
        x, y = get_my_batch(step)
        fingerprints.append(data_mod.batch_fingerprint(x))

        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                collectives.all_reduce(param.grad)
                param.grad /= world

        optimizer.step()
        losses.append(float(loss.detach()))

    return {"rank": rank, "losses": losses, "fingerprints": fingerprints[:5]}


def main() -> int:
    print(__doc__.split("Your job")[0])
    print("training 4 ranks for 30 steps ...\n")

    results = dist_utils.run_distributed(worker, world_size=4, backend="gloo", timeout_s=120)

    losses = results[0]["losses"]
    print(
        f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}   "
        f"(fell by {(1 - losses[-1] / losses[0]):.0%})\n"
    )

    print("  now hash each rank's first five batches:\n")
    print(f"    {'step':>6}  " + "  ".join(f"rank {r}" for r in sorted(results)))
    for step in range(5):
        row = "  ".join(f"{results[r]['fingerprints'][step][:6]}" for r in sorted(results))
        print(f"    {step:>6}  {row}")

    step0 = {results[r]["fingerprints"][0] for r in results}
    print()

    if len(step0) == 1:
        print("=" * 70)
        print("EVERY RANK IS TRAINING ON THE SAME DATA.")
        print("=" * 70)
        print("\n  Four ranks. One batch, replicated four times. The all-reduce")
        print("  averages four identical gradients, which gives you back exactly")
        print("  the gradient one rank would have computed alone.")
        print("\n  So this is a single-process run with four times the electricity")
        print("  and a quarter of the effective dataset. It converges faster on the")
        print("  training set because it is seeing the same examples over and over,")
        print("  and then it overfits, and you blame the model.")
        print("\n  Hypothesis in POSTMORTEM.md, then find it.")
        return 1

    print("  ranks see different data. Fixed. Now run:")
    print("    python -m pytest labs/lab_hazards/hazard_5_duplicate_data/test_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
