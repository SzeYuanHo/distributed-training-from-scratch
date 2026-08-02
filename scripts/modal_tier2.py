#!/usr/bin/env python3
"""Optionally run one lab's Tier-2 tests on a two-GPU Modal worker.

Nothing in the curriculum requires this provider.  This command creates
billable compute, so check Modal's current pricing and configure a spend budget
before using it.

Pass the public clone URL explicitly; the repository does not assume a GitHub
account name:

    pip install modal
    modal setup
    modal run scripts/modal_tier2.py \
        --repo https://github.com/OWNER/REPOSITORY.git --lab 1

By default the command overlays the published answer for the selected lab in a
disposable clone before testing it.  Pass ``--no-answers`` to test the starter
tree from the selected repository (for example, a fork containing your work).

The Tier-2 pytest cases call ``common.dist_utils.run_distributed`` themselves.
They must run as plain pytest; wrapping pytest or a bundled benchmark in
``torchrun`` would create a broken nested process topology.

Only Labs 0, 1, 2 and 3 currently define Tier-2 pytest cases. Lab 1B and Lab 4
have useful GPU benchmarks, but they are not accepted here as Tier-2 test jobs.
"""

from __future__ import annotations

import sys

try:
    import modal
except ImportError:  # pragma: no cover
    print(
        "modal is not installed. This script is optional.\n"
        "  pip install modal && modal setup\n"
        "For another optional hosted path, see:\n"
        "  notebooks/kaggle_2xT4_tier2.ipynb"
    )
    sys.exit(1)


GPU_SPEC = "A10:2"
TIMEOUT_S = 1_200
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"

# Match the Python 3.12 branches of requirements/cu126.lock.  The repository
# itself is installed from the requested clone with --no-deps below.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.13.0+cu126",
        "numpy==2.5.1",
        "pyyaml==6.0.3",
        "matplotlib==3.11.1",
        "psutil==7.2.2",
        "pytest==9.1.1",
        "pytest-timeout==2.4.0",
        "setuptools==83.0.0",
        extra_index_url=CUDA_INDEX,
    )
)

app = modal.App("dist-train-from-scratch-tier2", image=image)

LAB_DIRS = {
    "0": "lab0_collectives",
    "1": "lab1_ddp_bucketing",
    "2": "lab2_tensor_parallel",
    "3": "lab3_pipeline_parallel",
}


@app.function(gpu=GPU_SPEC, timeout=TIMEOUT_S)
def run_tier2(repo_url: str, lab: str, answers: bool = True) -> str:
    """Clone a repository and run one lab's self-spawning Tier-2 pytest cases."""
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    lab_name = LAB_DIRS.get(lab.strip().lower())
    if lab_name is None:
        raise ValueError(f"unknown lab {lab!r}; choose from {sorted(LAB_DIRS)}")
    if not repo_url.startswith("https://"):
        raise ValueError("--repo must be an explicit HTTPS Git clone URL")

    with tempfile.TemporaryDirectory(prefix="dtfs-tier2-") as temporary:
        checkout = Path(temporary) / "repository"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(checkout)],
            check=True,
            timeout=180,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(checkout)],
            check=True,
            cwd=checkout,
            timeout=180,
        )

        if answers:
            source = checkout / "answers" / lab_name / "starter"
            target = checkout / "labs" / lab_name / "starter"
            if not source.is_dir():
                raise FileNotFoundError(
                    f"published answer is missing from {source.relative_to(checkout)}"
                )
            shutil.copytree(source, target, dirs_exist_ok=True)

        env = dict(os.environ)
        env["DTFS_REPO_ROOT"] = str(checkout)
        env["PYTHONPATH"] = str(checkout)
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"

        command = [
            sys.executable,
            "-m",
            "pytest",
            str(checkout / "labs" / lab_name),
            "-m",
            "tier2",
            "-v",
            "--timeout=900",
        ]
        result = subprocess.run(
            command,
            env=env,
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=1_050,
        )
        output = result.stdout + "\n--- stderr ---\n" + result.stderr
        if result.returncode != 0:
            raise RuntimeError(f"pytest exited with status {result.returncode}\n{output}")
        return output


@app.local_entrypoint()
def main(repo: str, lab: str = "1", answers: bool = True):
    print(f"running lab {lab} Tier-2 tests on {GPU_SPEC}")
    print(f"published answers: {'yes' if answers else 'no'}")
    print(run_tier2.remote(repo, lab, answers))
