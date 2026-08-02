#!/usr/bin/env python3
"""Confirm nobody edited the frozen reference model.

``common/reference_gpt.py`` is the ground truth every lab compares against. The
moment someone tweaks it to make their tensor-parallel version match, every
equivalence test in the repo becomes a tautology. This hashes the file and
compares it against the recorded value.

The temptation is real and it looks reasonable at the time. You are three hours
into Lab 2, your logits are off by 1e-3, and changing the reference's GELU to
``approximate="tanh"`` makes it green. Do not. The mismatch is telling you
something about your sharding.

::

    python tasks.py freeze-check
    python scripts/freeze_check.py --record   # after a deliberate change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

FROZEN = ["common/reference_gpt.py"]


def manifest_path() -> Path:
    return repo_env.repo_root() / "scripts" / "frozen_files.json"


def digest(path: Path) -> str:
    # Normalise line endings so a Windows checkout does not fail a Linux hash.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--record", action="store_true", help="write the current hashes as the new baseline"
    )
    args = parser.parse_args(argv)

    root = repo_env.repo_root()
    missing_files = [rel for rel in FROZEN if not (root / rel).is_file()]
    if missing_files:
        print("\n  FROZEN FILE MISSING:\n", file=sys.stderr)
        for rel in missing_files:
            print(f"    {rel}", file=sys.stderr)
        print(
            "\n  Restore every protected file before checking or recording a baseline.",
            file=sys.stderr,
        )
        return 1

    current = {rel: digest(root / rel) for rel in FROZEN}

    if args.record:
        manifest_path().write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"  recorded {len(current)} hash(es) in {manifest_path().name}")
        for rel, sha in current.items():
            print(f"    {rel}  {sha[:16]}...")
        return 0

    manifest = manifest_path()
    if not manifest.is_file():
        print(
            f"\n  FROZEN BASELINE MISSING:\n\n"
            f"    {manifest}\n\n"
            "  This guard never creates a baseline implicitly. Restore the committed\n"
            "  manifest, or deliberately create one after review with:\n"
            "    python scripts/freeze_check.py --record\n",
            file=sys.stderr,
        )
        return 1

    try:
        baseline = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"\n  could not read frozen baseline {manifest}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(baseline, dict) or not all(
        isinstance(rel, str) and isinstance(sha, str) for rel, sha in baseline.items()
    ):
        print(f"\n  invalid frozen baseline format in {manifest}", file=sys.stderr)
        return 1

    expected = set(FROZEN)
    recorded = set(baseline)
    missing_records = sorted(expected - recorded)
    stale_records = sorted(recorded - expected)
    invalid_hashes = sorted(
        rel
        for rel in expected & recorded
        if len(baseline[rel]) != 64
        or any(char not in "0123456789abcdef" for char in baseline[rel].lower())
    )
    if missing_records or stale_records or invalid_hashes:
        print("\n  FROZEN BASELINE INVALID:\n", file=sys.stderr)
        for rel in missing_records:
            print(f"    missing record: {rel}", file=sys.stderr)
        for rel in stale_records:
            print(f"    stale record:   {rel}", file=sys.stderr)
        for rel in invalid_hashes:
            print(f"    invalid hash:   {rel}", file=sys.stderr)
        print(
            "\n  Restore the committed manifest, or deliberately replace it after review\n"
            "  with: python scripts/freeze_check.py --record\n",
            file=sys.stderr,
        )
        return 1

    changed = [rel for rel in FROZEN if baseline[rel].lower() != current[rel]]
    if not changed:
        print(f"  {len(current)} frozen file(s) unchanged.")
        return 0

    print("\n  FROZEN FILE MODIFIED:\n")
    for rel in changed:
        print(f"    {rel}")
        print(f"      expected {baseline[rel][:16]}...")
        print(f"      found    {current[rel][:16]}...")
    print(
        "\n  Every lab compares against this model. Editing it makes those comparisons\n"
        "  meaningless. Restore the reviewed version from source control:\n"
        f"    git restore -- {' '.join(changed)}\n"
        "\n  If the change really is intended, record the new baseline:\n"
        "    python scripts/freeze_check.py --record\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
