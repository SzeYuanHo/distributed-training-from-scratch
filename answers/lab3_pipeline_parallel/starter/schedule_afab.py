from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from . import p2p

__all__ = ["Op", "afab_schedule", "run_afab", "max_inflight"]


@dataclass(frozen=True)
class Op:
    kind: str
    micro: int

    def __str__(self) -> str:
        return f"{self.kind}{self.micro}"


def afab_schedule(stage_index: int, n_stages: int, n_micro: int) -> list[Op]:
    del stage_index, n_stages
    return [Op("F", i) for i in range(n_micro)] + [Op("B", i) for i in reversed(range(n_micro))]


def max_inflight(schedule: list[Op]) -> int:
    current = peak = 0
    for op in schedule:
        current += 1 if op.kind == "F" else -1
        peak = max(peak, current)
    return peak


def _span(event_log, label: str):
    return event_log.span(label, category="compute") if event_log else contextlib.nullcontext()


def _input_shape(stage, batch: torch.Tensor) -> tuple[int, int, int]:
    return stage.activation_shape(batch.size(0), batch.size(1), stage.hidden_size)


def run_afab(
    stage, n_micro: int, batches: list, targets: list | None = None, event_log=None
) -> float:
    previous, following = p2p.send_recv_pairs(stage.stage_index, stage.n_stages)
    dtype = next(stage.parameters()).dtype
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    losses: dict[int, torch.Tensor] = {}
    total_loss = 0.0

    for micro in range(n_micro):
        if stage.is_first:
            value = batches[micro]
        else:
            value = p2p.recv_activation(
                _input_shape(stage, batches[micro]), dtype, previous, requires_grad=True
            )
        with _span(event_log, f"F{micro}"):
            result = stage(value, targets[micro] if stage.is_last else None)
        inputs[micro] = value
        if stage.is_last:
            logits, loss = result
            outputs[micro] = logits
            losses[micro] = loss
            total_loss += float(loss.detach()) / n_micro
        else:
            outputs[micro] = result
            p2p.send_activation(result, following)

    for micro in reversed(range(n_micro)):
        with _span(event_log, f"B{micro}"):
            if stage.is_last:
                (losses[micro] / n_micro).backward()
            else:
                grad = p2p.recv_activation(
                    tuple(outputs[micro].shape),
                    outputs[micro].dtype,
                    following,
                    requires_grad=False,
                )
                outputs[micro].backward(grad)
        if not stage.is_first:
            p2p.send_activation(inputs[micro].grad, previous)

    return total_loss
