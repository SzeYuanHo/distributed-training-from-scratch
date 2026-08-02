# shellcheck shell=bash
#
# Source this, do not execute it:
#
#     source activate.sh
#
# It activates ./.venv and redirects the managed workflow's known third-party
# caches into ./.cache. Sourcing twice is harmless. The same variables are set
# programmatically by common/repo_env.py, so `make` targets work whether or not
# you have sourced this.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _DTFS_SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _DTFS_SELF="${(%):-%x}"
else
    _DTFS_SELF="$0"
fi

REPO_ROOT="$(cd "$(dirname "$_DTFS_SELF")" && pwd)"
export REPO_ROOT
export DTFS_REPO_ROOT="$REPO_ROOT"

# --- virtualenv ------------------------------------------------------------
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/.venv/Scripts/activate" ]; then
    # Git Bash on Windows lays the venv out the Windows way.
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.venv/Scripts/activate"
else
    echo "note: no .venv yet. Run 'python tasks.py setup' (or 'make setup')."
fi

# --- cache redirection -----------------------------------------------------
CACHE="$REPO_ROOT/.cache"
export DTFS_CACHE_ACTIVE=1
export XDG_CACHE_HOME="$CACHE"
export HF_HOME="$CACHE/huggingface"
export HF_DATASETS_CACHE="$CACHE/huggingface/datasets"
export TORCH_HOME="$CACHE/torch"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PIP_CACHE_DIR="$CACHE/pip"
export UV_CACHE_DIR="$CACHE/uv"
export MPLCONFIGDIR="$CACHE/matplotlib"
export WANDB_DIR="$CACHE/wandb"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

mkdir -p "$CACHE/huggingface" "$CACHE/torch" "$CACHE/triton" "$CACHE/inductor" \
         "$CACHE/pip" "$CACHE/uv" "$CACHE/matplotlib" "$CACHE/wandb"

# --- CPU thread budget -----------------------------------------------------
# Each rank otherwise grabs every core. Four ranks on sixteen cores means
# sixty-four OpenMP threads fighting over sixteen, and everything crawls.
# Set DTFS_NPROC before sourcing if you know how many ranks you will launch.
_DTFS_CORES="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
_DTFS_NPROC="${DTFS_NPROC:-2}"
case "$_DTFS_NPROC" in
    ""|*[!0-9]*|0)
        echo "warning: DTFS_NPROC must be a positive integer; using 2."
        _DTFS_NPROC=2
        ;;
esac
export DTFS_NPROC="$_DTFS_NPROC"
_DTFS_THREADS=$(( _DTFS_CORES / _DTFS_NPROC ))
[ "$_DTFS_THREADS" -lt 1 ] && _DTFS_THREADS=1
export OMP_NUM_THREADS="$_DTFS_THREADS"
export MKL_NUM_THREADS="$_DTFS_THREADS"

echo "dist-train-from-scratch"
echo "  repo    $REPO_ROOT"
echo "  caches  $CACHE  (managed third-party caches)"
echo "  threads OMP_NUM_THREADS=$OMP_NUM_THREADS for DTFS_NPROC=$_DTFS_NPROC of $_DTFS_CORES cores"
echo "  next    python tasks.py doctor"

unset _DTFS_SELF _DTFS_CORES _DTFS_NPROC _DTFS_THREADS
