# Dependency locks

The project metadata in `pyproject.toml` defines supported version ranges. These generated
lock files define the exact environments used by the managed setup commands:

- `cpu.lock` uses the CPU-only PyTorch index and is the default for Tier 0/1 and CI.
- `cu126.lock` uses the CUDA 12.6 PyTorch index for the optional local GPU profile.

Regenerate them with the tested uv release:

```bash
uv pip compile pyproject.toml --extra dev --universal --python-version 3.10 \
  --torch-backend cpu --generate-hashes --output-file requirements/cpu.lock
uv pip compile pyproject.toml --extra dev --universal --python-version 3.10 \
  --torch-backend cu126 --generate-hashes --output-file requirements/cu126.lock
```

The lock files cover project, development and pinned build-backend dependencies. The
editable repository itself is installed afterward with `--no-deps
--no-build-isolation`, so neither dependency resolution nor an isolated build
environment can introduce unpinned packages.

Managed setup also passes `--index-strategy unsafe-best-match`: PyPI supplies general
dependencies while the PyTorch index supplies backend-specific wheels, and uv's default
first-index strategy stops at duplicate support packages on the PyTorch index. This does
not make the accepted artefact set open-ended because `--require-hashes` still restricts
every installation to the hashes in the selected lock.
