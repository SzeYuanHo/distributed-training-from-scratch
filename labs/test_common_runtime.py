"""Regression tests for shared runtime configuration."""

from __future__ import annotations

import pytest
import torch

from common import dist_utils, repo_env
from common.testing import deterministic_context
from common.timers import Timer


@pytest.mark.tier0
def test_device_for_rank_keeps_initialized_gloo_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist_utils, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dist_utils.torch.cuda, "device_count", lambda: 8)

    assert dist_utils.device_for_rank().type == "cpu"


@pytest.mark.tier0
@pytest.mark.parametrize("value", ["two", "0", "-1"])
def test_positive_integer_environment_values_are_validated(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DTFS_TEST_POSITIVE_INT", value)

    with pytest.raises(ValueError, match="DTFS_TEST_POSITIVE_INT must be a positive integer"):
        repo_env.positive_int_env("DTFS_TEST_POSITIVE_INT", 2)


@pytest.mark.tier0
def test_apply_propagates_rank_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTFS_NPROC", raising=False)

    repo_env.apply(nproc=4, create=False)

    assert repo_env.positive_int_env("DTFS_NPROC", 2) == 4


@pytest.mark.tier0
def test_distributed_launcher_rejects_nonpositive_counts_and_timeouts() -> None:
    with pytest.raises(ValueError, match="world_size must be a positive integer"):
        dist_utils.pick_backend(0)
    with pytest.raises(ValueError, match="timeout_s must be a positive integer"):
        dist_utils.run_distributed(lambda: None, timeout_s=0)
    with pytest.raises(ValueError, match="join_timeout_s must be positive"):
        dist_utils.run_distributed(lambda: None, join_timeout_s=0)


@pytest.mark.tier0
def test_deterministic_context_restores_algorithm_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: calls.append((enabled, warn_only)),
    )

    with deterministic_context(warn_only=False):
        pass

    assert calls[-1] == (True, True)


@pytest.mark.tier0
def test_timer_report_does_not_restore_skipped_samples() -> None:
    timer = Timer(device="cpu")
    timer._samples["step"] = [12.5]

    report = timer.report(skip_first=1)

    assert timer.mean("step", skip_first=1) == 0.0
    assert "step                          0" in report
