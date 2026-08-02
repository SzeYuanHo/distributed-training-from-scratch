"""Regression tests for the optional-corpus byte cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from common import data as data_module


class _Response:
    def __init__(self, body: bytes, *, content_length: str | None = None) -> None:
        self._body = body
        self.read_sizes: list[int] = []
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk


@pytest.mark.tier0
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_Response(b"", content_length="11"), "response is 11 bytes"),
        (_Response(b"01234567890"), "response exceeds"),
    ],
)
def test_tinystories_rejects_oversized_response_without_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: _Response,
    message: str,
) -> None:
    monkeypatch.setenv("DTFS_ALLOW_TINYSTORIES", "1")
    monkeypatch.setattr(data_module, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match=message):
        data_module.ensure_tinystories(max_bytes=10)

    assert not (tmp_path / "tinystories_valid.txt").exists()
    assert not list(tmp_path.glob(".tinystories_valid.txt.*.tmp"))


@pytest.mark.tier0
def test_shakespeare_download_is_bounded_before_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = _Response(b"x" * 100)
    monkeypatch.setattr(data_module, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(data_module, "SHAKESPEARE_BYTES", 10)
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="response exceeds"):
        data_module.ensure_shakespeare()

    assert response.read_sizes == [11]
    assert not (tmp_path / "tinyshakespeare.txt").exists()


@pytest.mark.tier0
def test_tinystories_replaces_unmarked_partial_file_and_reuses_completed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "tinystories_valid.txt"
    target.write_bytes(b"x")
    response = _Response(b"complete", content_length="8")
    monkeypatch.setenv("DTFS_ALLOW_TINYSTORIES", "1")
    monkeypatch.setattr(data_module, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    assert data_module.ensure_tinystories(max_bytes=10).read_bytes() == b"complete"
    assert (tmp_path / "tinystories_valid.txt.complete.json").is_file()

    def unexpected_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a complete, verified cache should be reused")

    monkeypatch.setattr(data_module.urllib.request, "urlopen", unexpected_download)
    assert data_module.ensure_tinystories(max_bytes=10) == target
