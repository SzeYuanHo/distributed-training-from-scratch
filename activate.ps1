# Dot-source this, do not run it:
#
#     . .\activate.ps1
#
# Activates .\.venv and redirects the managed workflow's known third-party
# caches into .\.cache. Running it twice is fine. common/repo_env.py sets the
# same variables from Python, so the `make.ps1` targets work whether or not you
# dot-sourced this first.

$RepoRoot = $PSScriptRoot
$env:REPO_ROOT = $RepoRoot
$env:DTFS_REPO_ROOT = $RepoRoot

# --- virtualenv ------------------------------------------------------------
$activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
} else {
    Write-Host "note: no .venv yet. Run 'python tasks.py setup'."
}

# --- cache redirection -----------------------------------------------------
$Cache = Join-Path $RepoRoot ".cache"
$env:DTFS_CACHE_ACTIVE      = "1"
$env:XDG_CACHE_HOME         = $Cache
$env:HF_HOME                = Join-Path $Cache "huggingface"
$env:HF_DATASETS_CACHE      = Join-Path $Cache "huggingface\datasets"
$env:TORCH_HOME             = Join-Path $Cache "torch"
$env:TRITON_CACHE_DIR       = Join-Path $Cache "triton"
$env:TORCHINDUCTOR_CACHE_DIR= Join-Path $Cache "inductor"
$env:PIP_CACHE_DIR          = Join-Path $Cache "pip"
$env:UV_CACHE_DIR           = Join-Path $Cache "uv"
$env:MPLCONFIGDIR           = Join-Path $Cache "matplotlib"
$env:WANDB_DIR              = Join-Path $Cache "wandb"
$env:WANDB_MODE             = "offline"
$env:TOKENIZERS_PARALLELISM = "false"

foreach ($sub in @("huggingface","torch","triton","inductor","pip","uv","matplotlib","wandb")) {
    $p = Join-Path $Cache $sub
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

# --- CPU thread budget -----------------------------------------------------
# Without this, N ranks each start one OpenMP thread per core and the machine
# spends its time context-switching instead of computing.
$cores = [Environment]::ProcessorCount
$parsedNproc = 0
$validNproc = $env:DTFS_NPROC -and [int]::TryParse($env:DTFS_NPROC, [ref]$parsedNproc) -and $parsedNproc -gt 0
if ($validNproc) {
    $nproc = $parsedNproc
} else {
    if ($env:DTFS_NPROC) {
        Write-Warning "DTFS_NPROC must be a positive integer; using 2."
    }
    $nproc = 2
}
$env:DTFS_NPROC = "$nproc"
$threads = [Math]::Max(1, [Math]::Floor($cores / $nproc))
$env:OMP_NUM_THREADS = "$threads"
$env:MKL_NUM_THREADS = "$threads"

Write-Host "dist-train-from-scratch"
Write-Host "  repo    $RepoRoot"
Write-Host "  caches  $Cache  (managed third-party caches)"
Write-Host "  threads OMP_NUM_THREADS=$threads for DTFS_NPROC=$nproc of $cores cores"
Write-Host "  next    python tasks.py doctor"
Write-Host ""
Write-Host "  Native Windows runs Tiers 0 and 1 on gloo. Tier 2 needs a Linux"
Write-Host "  environment with at least two visible GPUs, when one is available."
