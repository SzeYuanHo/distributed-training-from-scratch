#!/usr/bin/env python3
"""Assert every lab still has the eight artefacts the modularity contract requires.

Labs drift. Someone adds a lab without a THEORY.md, or deletes a
predictions.yaml because it was empty anyway, and six months later the repo has
seven labs with seven different shapes. This is the cheap guard against that.

The public lab contract is::

    README.md  THEORY.md  predictions.yaml  starter/  reference/  tests/  bench/  SOLUTION.md

``lab_hazards`` is exempt: five broken scripts is a deliberately different
shape, and it gets its own check. The published ``answers/`` mirror is checked
at the same time so CI cannot silently certify an incomplete reference set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

REQUIRED = [
    "README.md",
    "THEORY.md",
    "predictions.yaml",
    "starter",
    "reference",
    "tests",
    "bench",
    "SOLUTION.md",
]

REQUIRED_TESTS = ["test_00_math.py", "test_01_semantics.py"]

HAZARD_FILES = ["broken.py", "HINT.md", "POSTMORTEM.md"]

EXPECTED_LABS = {
    "lab0_collectives",
    "lab1_ddp_bucketing",
    "lab1b_zero_memory",
    "lab2_tensor_parallel",
    "lab3_pipeline_parallel",
    "lab4_capstone_2d",
    "lab_hazards",
}

EXPECTED_HAZARDS = {
    "hazard_1_the_hang",
    "hazard_2_the_desync",
    "hazard_3_silent_gradient",
    "hazard_4_pipeline_deadlock",
    "hazard_5_duplicate_data",
}

README_SECTIONS = [
    "Objectives",
    "Time & tier budget",
    "Required reading",
    "The math",
    "Build steps",
    "Verification",
    "Instrumentation",
    "Reconciliation",
    "Stretch goals",
    "Pitfalls",
    "Where this lives in production",
]


def check_lab(lab_dir: Path) -> list[str]:
    problems = []
    for artefact in REQUIRED:
        if not (lab_dir / artefact).exists():
            problems.append(f"missing {artefact}")

    tests = lab_dir / "tests"
    if tests.exists():
        for name in REQUIRED_TESTS:
            if not (tests / name).exists():
                problems.append(f"missing tests/{name}")

    readme = lab_dir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        missing = [s for s in README_SECTIONS if s.lower() not in text.lower()]
        if missing:
            problems.append(f"README is missing sections: {', '.join(missing)}")

    starter = lab_dir / "starter"
    if starter.exists():
        stubs = list(starter.glob("*.py"))
        if not stubs:
            problems.append("starter/ has no python files")
        elif not any("NotImplementedError" in s.read_text(encoding="utf-8") for s in stubs):
            problems.append(
                "no starter file raises NotImplementedError; exercise stubs must stay incomplete"
            )

    return problems


def check_hazards(hazard_root: Path) -> list[str]:
    problems = []
    dirs = sorted(d for d in hazard_root.iterdir() if d.is_dir() and d.name.startswith("hazard_"))
    names = {d.name for d in dirs}
    missing = sorted(EXPECTED_HAZARDS - names)
    unexpected = sorted(names - EXPECTED_HAZARDS)
    if missing:
        problems.append(f"missing hazard directories: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected hazard directories: {', '.join(unexpected)}")
    for d in dirs:
        for name in HAZARD_FILES:
            if not (d / name).exists():
                problems.append(f"{d.name}: missing {name}")
    return problems


def check_published_answer(root: Path, lab_dir: Path) -> list[str]:
    answer = root / "answers" / lab_dir.name
    problems: list[str] = []

    if not answer.is_dir():
        return ["published answer directory is missing"]

    if lab_dir.name == "lab_hazards":
        hazard_dirs = {
            d.name for d in answer.iterdir() if d.is_dir() and d.name.startswith("hazard_")
        }
        missing = sorted(EXPECTED_HAZARDS - hazard_dirs)
        unexpected = sorted(hazard_dirs - EXPECTED_HAZARDS)
        if missing:
            problems.append(f"published answer missing hazard directories: {', '.join(missing)}")
        if unexpected:
            problems.append(
                f"published answer has unexpected hazard directories: {', '.join(unexpected)}"
            )
        for hazard in sorted(
            d for d in lab_dir.iterdir() if d.is_dir() and d.name.startswith("hazard_")
        ):
            for name in ("broken.py", "POSTMORTEM.md"):
                if not (answer / hazard.name / name).is_file():
                    problems.append(f"published answer missing {hazard.name}/{name}")
        return problems

    exercise_starter = lab_dir / "starter"
    answer_starter = answer / "starter"
    if not answer_starter.is_dir():
        problems.append("published answer missing starter/")
    else:
        for exercise_file in exercise_starter.glob("*.py"):
            completed_file = answer_starter / exercise_file.name
            if not completed_file.is_file():
                problems.append(f"published answer missing starter/{exercise_file.name}")
                continue
            if "NotImplementedError" in completed_file.read_text(encoding="utf-8"):
                problems.append(f"published answer starter/{exercise_file.name} is incomplete")

    for artefact in ("THEORY.md", "predictions.yaml", "runs"):
        if not (answer / artefact).exists():
            problems.append(f"published answer missing {artefact}")

    return problems


def main() -> int:
    root = repo_env.repo_root()
    labs = root / "labs"
    if not labs.exists():
        print("  labs/ does not exist yet.")
        return 1
    answers = root / "answers"
    workspace = root / "workspace"

    def is_lab(path: Path) -> bool:
        # Skip __pycache__, .pytest_cache and anything else that is not a lab.
        return path.is_dir() and not path.name.startswith((".", "_"))

    failures = 0
    actual_labs = {d.name for d in labs.iterdir() if is_lab(d)}
    actual_answers = {d.name for d in answers.iterdir() if is_lab(d)} if answers.is_dir() else set()
    actual_workspace = (
        {d.name for d in workspace.iterdir() if is_lab(d)} if workspace.is_dir() else set()
    )
    repository_problems: list[str] = []
    for label, actual in (
        ("labs", actual_labs),
        ("answers", actual_answers),
        ("workspace", actual_workspace),
    ):
        missing = sorted(EXPECTED_LABS - actual)
        unexpected = sorted(actual - EXPECTED_LABS)
        if missing:
            repository_problems.append(f"{label}/ missing: {', '.join(missing)}")
        if unexpected:
            repository_problems.append(f"{label}/ unexpected: {', '.join(unexpected)}")

    if repository_problems:
        failures += 1
        print("\n  repository mirrors")
        for problem in repository_problems:
            print(f"    - {problem}")

    for name in sorted(EXPECTED_LABS & actual_labs):
        lab_dir = labs / name
        if lab_dir.name == "lab_hazards":
            problems = check_hazards(lab_dir)
        else:
            problems = check_lab(lab_dir)
        problems.extend(check_published_answer(root, lab_dir))

        if problems:
            failures += 1
            print(f"\n  {lab_dir.name}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"  {lab_dir.name}: ok")

    if failures:
        print(f"\n  {failures} lab(s) do not match the public structure contract.\n")
        return 1

    print("\n  Every lab matches the structure contract.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
