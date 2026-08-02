"""tiny-shakespeare, a character tokenizer, and rank-disjoint batching.

1.1 MB of text, 65 distinct characters, and it trains a small GPT to something
readable in a couple of minutes. Nothing about the curriculum depends on the
data being interesting.

The part worth reading is :func:`get_batch`, and specifically the property it
guarantees:

    For any step, the union of the batches drawn by ranks 0..N-1 equals the
    single-process batch of N times the size drawn at that same step.

Lab 1's correctness test depends on that. If it did not hold, a 2-rank DDP run
would be looking at different data from the 1-rank reference and their loss
curves could never match, and you would spend an evening debugging your
all-reduce when the bug was in the data loader.

It holds by construction rather than by luck: the function builds the *global*
batch first, deterministically from ``(seed, step)``, then hands rank ``r`` its
contiguous slice. There is no per-rank randomness to get wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "SHAKESPEARE_URL",
    "SHAKESPEARE_SHA256",
    "CharTokenizer",
    "Dataset",
    "ensure_shakespeare",
    "load_dataset",
    "get_batch",
    "dataset_stats",
]

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Measured against the canonical file: 1,115,394 bytes, 65 distinct characters.
# A mismatch means upstream changed the file or the download was truncated.
# Exact cross-machine comparisons require this precise corpus.
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
SHAKESPEARE_BYTES = 1_115_394

# Padding the vocabulary so it divides by the maximum tensor-parallel degree.
# The real character set is 65 entries; the extra rows never appear in the data
# and train to never being predicted.
VOCAB_MULTIPLE = 8


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


class CharTokenizer:
    """Character-level vocabulary built from the corpus itself.

    Deterministic: the vocabulary is the sorted set of characters present, so
    two processes that read the same file get the same mapping without having
    to share anything.
    """

    def __init__(self, text: str, pad_to_multiple: int = VOCAB_MULTIPLE):
        chars = sorted(set(text))
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.raw_vocab_size = len(chars)

        padded = self.raw_vocab_size
        if pad_to_multiple > 1 and padded % pad_to_multiple:
            padded += pad_to_multiple - (padded % pad_to_multiple)
        self.vocab_size = padded

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) < self.raw_vocab_size)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"<CharTokenizer {self.raw_vocab_size} chars, "
            f"padded to {self.vocab_size} for TP divisibility>"
        )


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    train: np.ndarray  # uint16 token ids
    val: np.ndarray
    tokenizer: CharTokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def split(self, name: str) -> np.ndarray:
        if name not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val'; got {name!r}")
        return self.train if name == "train" else self.val


def data_dir() -> Path:
    from common import repo_env

    path = repo_env.repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _shakespeare_mismatch(payload_size: int, payload_sha256: str) -> str | None:
    problems = []
    if payload_size != SHAKESPEARE_BYTES:
        problems.append(f"size {payload_size:,} bytes (expected {SHAKESPEARE_BYTES:,})")
    if payload_sha256 != SHAKESPEARE_SHA256:
        problems.append(f"SHA-256 {payload_sha256[:16]}... (expected {SHAKESPEARE_SHA256[:16]}...)")
    return "; ".join(problems) if problems else None


def ensure_shakespeare(*, verify: bool = True, quiet: bool = False) -> Path:
    """Download tiny-shakespeare into ``data/`` if it is not already there.

    Returns the path to the raw text. Roughly 1.1 MB, one HTTP request, and it
    happens once. ``python tasks.py setup`` calls this so a fresh clone is ready
    to run. Existing and downloaded files are verified by default. Passing
    ``verify=False`` is the explicit opt-out for a deliberately different
    corpus; exact reference-loss comparisons then no longer apply.
    """
    target = data_dir() / "tinyshakespeare.txt"
    existing_problem = None
    if target.is_file():
        if not verify:
            return target
        existing_problem = _shakespeare_mismatch(target.stat().st_size, _file_sha256(target))
        if existing_problem is None:
            return target

    if not quiet:
        if existing_problem:
            print(f"  existing tiny-shakespeare is invalid: {existing_problem}")
        print(f"  downloading tiny-shakespeare (1.1 MB) -> {target}")

    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
            payload = response.read(SHAKESPEARE_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise RuntimeError(
            f"could not download tiny-shakespeare from {SHAKESPEARE_URL}\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"  If you are offline, drop the file at {target} yourself. It is the\n"
            f"  plain-text Shakespeare corpus from Karpathy's char-rnn repository."
        ) from exc

    if len(payload) > SHAKESPEARE_BYTES:
        raise RuntimeError(
            "tiny-shakespeare response exceeds the recorded "
            f"{SHAKESPEARE_BYTES:,}-byte corpus size. No file was written.\n"
            f"  source: {SHAKESPEARE_URL}"
        )

    digest = hashlib.sha256(payload).hexdigest()
    mismatch = _shakespeare_mismatch(len(payload), digest)
    if verify and mismatch:
        raise RuntimeError(
            "downloaded tiny-shakespeare does not match the recorded corpus:\n"
            f"  {mismatch}\n"
            f"  source: {SHAKESPEARE_URL}\n"
            "  The existing file, if any, was left untouched. If you deliberately\n"
            "  want a different corpus, call ensure_shakespeare(verify=False); exact\n"
            "  reference-loss comparisons will no longer apply."
        )

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    if not quiet:
        print(f"  wrote {len(payload) / 1e6:.2f} MB")
    return target


_CACHE: dict[tuple[float, str], Dataset] = {}


def load_dataset(val_fraction: float = 0.1, *, use_cache: bool = True) -> Dataset:
    """Tokenize the corpus and split it. Cached per process.

    Tokens are stored as ``uint16`` and memory-mapped. At this corpus size the
    memmap buys nothing, but it is the same code path you would use for a real
    dataset, and Lab 1B measures memory where the difference would show.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1; got {val_fraction}")

    raw = ensure_shakespeare(quiet=True)
    source_sha256 = _file_sha256(raw)
    key = (val_fraction, source_sha256)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    text = raw.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    # Content-address the token cache so changing the source corpus can never
    # silently reuse tokens produced from an earlier file.
    bin_path = data_dir() / f"tinyshakespeare.{source_sha256[:16]}.uint16.npy"
    regenerate = True
    if bin_path.is_file():
        try:
            tokens = np.load(bin_path, mmap_mode="r")
            regenerate = tokens.dtype != np.uint16 or len(tokens) != len(text)
        except (OSError, ValueError):
            regenerate = True
    if regenerate:
        tokens = np.array(tokenizer.encode(text), dtype=np.uint16)
        temporary = bin_path.with_name(f".{bin_path.name}.{os.getpid()}.tmp.npy")
        try:
            np.save(temporary, tokens)
            temporary.replace(bin_path)
        finally:
            temporary.unlink(missing_ok=True)
        tokens = np.load(bin_path, mmap_mode="r")

    split_at = int(len(tokens) * (1 - val_fraction))
    dataset = Dataset(train=tokens[:split_at], val=tokens[split_at:], tokenizer=tokenizer)

    if use_cache:
        _CACHE[key] = dataset
    return dataset


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #


def get_batch(
    split: str = "train",
    batch_size: int = 8,
    block_size: int = 128,
    *,
    rank: int = 0,
    world_size: int = 1,
    step: int = 0,
    seed: int = 1337,
    device: torch.device | str | None = None,
    dataset: Dataset | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, rank-disjoint batch of ``(x, y)`` with shape ``(batch_size, block_size)``.

    ``batch_size`` is the **per-rank** size. Total tokens per step across the
    job is ``batch_size * world_size * block_size``.

    The union property this guarantees:

    >>> single = get_batch("train", batch_size=8, world_size=1, step=5)
    >>> r0 = get_batch("train", batch_size=4, rank=0, world_size=2, step=5)
    >>> r1 = get_batch("train", batch_size=4, rank=1, world_size=2, step=5)
    >>> torch.cat([r0[0], r1[0]]).equal(single[0])
    True

    That is why Lab 1 can compare a 2-rank DDP loss curve against a 1-rank
    reference and expect them to agree.

    Implementation: seed a generator from ``(seed, step)``, draw
    ``batch_size * world_size`` starting offsets in one go, then slice out
    rank ``r``'s contiguous block. No rank ever draws its own randomness.

    On the ragged tail
    ------------------
    There is none here, deliberately. Offsets are sampled with replacement from
    the whole corpus rather than walked in order, so every rank always has a
    batch and no rank ever runs out first. A real loader that iterates an epoch
    has to either drop the remainder or pad it, because a rank that stops
    calling collectives while its peers keep going produces a hang at the end
    of every epoch. That failure is Lab H hazard 5's cousin, and it is worth
    knowing it exists even though this loader sidesteps it.
    """
    ds = dataset if dataset is not None else load_dataset()
    tokens = ds.split(split)

    n_total = batch_size * world_size
    max_start = len(tokens) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"block_size {block_size} is too large for the {split} split ({len(tokens)} tokens)."
        )

    # One generator, one draw, identical on every rank because the seed depends
    # only on (seed, step). Mixing in the step keeps successive batches
    # different without any hidden state.
    generator = np.random.default_rng(seed * 1_000_003 + step)
    starts = generator.integers(0, max_start, size=n_total, dtype=np.int64)

    lo = rank * batch_size
    my_starts = starts[lo : lo + batch_size]

    x = np.stack([tokens[s : s + block_size] for s in my_starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + 1 + block_size] for s in my_starts]).astype(np.int64)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device is not None:
        xt = xt.to(device, non_blocking=True)
        yt = yt.to(device, non_blocking=True)
    return xt, yt


def batch_fingerprint(x: torch.Tensor) -> str:
    """Stable hash of a batch. Lab H hazard 5 uses it to prove ranks differ.

    Hashing the tensor rather than comparing it means you can gather one short
    string per rank instead of the whole batch.
    """
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def dataset_stats(dataset: Dataset | None = None) -> str:
    ds = dataset if dataset is not None else load_dataset()
    return (
        f"tiny-shakespeare\n"
        f"  train tokens   {len(ds.train):,}\n"
        f"  val tokens     {len(ds.val):,}\n"
        f"  characters     {ds.tokenizer.raw_vocab_size} "
        f"(padded to {ds.vocab_size} for TP divisibility)\n"
        f"  on disk        {(data_dir() / 'tinyshakespeare.txt').stat().st_size / 1e6:.2f} MB"
    )


# --------------------------------------------------------------------------- #
# the optional larger corpus
# --------------------------------------------------------------------------- #

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
TINYSTORIES_CAP_BYTES = 500 * 1024 * 1024


def _tinystories_marker(target: Path) -> Path:
    return target.with_name(f"{target.name}.complete.json")


def _tinystories_cache_is_complete(target: Path, marker: Path, max_bytes: int) -> bool:
    if not target.is_file() or not marker.is_file():
        return False
    size = target.stat().st_size
    if not 0 < size <= max_bytes:
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        metadata.get("url") == TINYSTORIES_URL
        and metadata.get("size") == size
        and metadata.get("sha256") == _file_sha256(target)
    )


def ensure_tinystories(max_bytes: int = TINYSTORIES_CAP_BYTES) -> Path:
    """Optional larger corpus, behind an environment flag and a hard byte cap.

    Nothing in the curriculum needs this. It exists so that if you want to see
    whether your conclusions survive a corpus that does not fit in L3 cache,
    you can, without accidentally filling a 50 GB disk.
    """
    if os.environ.get("DTFS_ALLOW_TINYSTORIES", "0") != "1":
        raise RuntimeError(
            "TinyStories is off by default because it is a few hundred megabytes.\n"
            "  Set DTFS_ALLOW_TINYSTORIES=1 if you want it. Nothing in the labs "
            "requires it, and tiny-shakespeare keeps Tier-1 runs to seconds."
        )

    if max_bytes < 1:
        raise ValueError(f"max_bytes must be positive; got {max_bytes}")

    target = data_dir() / "tinystories_valid.txt"
    marker = _tinystories_marker(target)
    if _tinystories_cache_is_complete(target, marker, max_bytes):
        return target
    if target.is_file() and target.stat().st_size > max_bytes:
        raise RuntimeError(
            f"{target} is {target.stat().st_size:,} bytes, over the requested "
            f"{max_bytes:,}-byte cap. Remove or relocate it before downloading "
            "a capped copy."
        )
    if target.is_file():
        print("  existing TinyStories file has no valid completion marker; replacing it")

    print(f"  downloading TinyStories (limit {max_bytes / 1e6:.0f} MB)")
    written = 0
    digest = hashlib.sha256()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    marker_temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with (
            urllib.request.urlopen(TINYSTORIES_URL, timeout=300) as response,
            temporary.open("wb") as out,
        ):
            content_length = response.headers.get("Content-Length")
            declared_size = None
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise RuntimeError(
                        f"TinyStories response is {declared_size:,} bytes, over the "
                        f"configured {max_bytes:,}-byte limit. No file was written."
                    )

            while True:
                # Once the limit is reached, read one more byte. A non-empty
                # result proves the response is oversized; never publish a
                # silently truncated corpus as if it were complete.
                remaining = max_bytes - written
                chunk = response.read(min(1 << 20, remaining + 1))
                if not chunk:
                    break
                if len(chunk) > remaining:
                    raise RuntimeError(
                        f"TinyStories response exceeds the configured "
                        f"{max_bytes:,}-byte limit. No file was written."
                    )
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)

        if written == 0:
            raise RuntimeError("TinyStories response was empty. No file was written.")
        if declared_size is not None and written != declared_size:
            raise RuntimeError(
                f"TinyStories response ended after {written:,} bytes, but its "
                f"Content-Length declared {declared_size:,}. No file was written."
            )

        marker_temporary.write_text(
            json.dumps(
                {
                    "url": TINYSTORIES_URL,
                    "size": written,
                    "sha256": digest.hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        marker_temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
        marker_temporary.unlink(missing_ok=True)
    print(f"  wrote {written / 1e6:.0f} MB to {target}")
    return target
