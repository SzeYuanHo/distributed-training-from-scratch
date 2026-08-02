from __future__ import annotations

import contextlib

import torch

from . import p2p
from .schedule_afab import Op

__all__ = ["onefonebe_schedule", "warmup_steps", "run_1f1b"]


def warmup_steps(stage_index: int, n_stages: int, n_micro: int) -> int:
    return min(n_stages - stage_index, n_micro)


def onefonebe_schedule(stage_index: int, n_stages: int, n_micro: int) -> list[Op]:
    warmup = warmup_steps(stage_index, n_stages, n_micro)
    ops = [Op("F", i) for i in range(warmup)]
    forward, backward = warmup, 0
    while forward < n_micro:
        ops.append(Op("B", backward))
        ops.append(Op("F", forward))
        forward += 1
        backward += 1
    ops.extend(Op("B", i) for i in range(backward, n_micro))
    return ops


def _span(event_log, label: str):
    return event_log.span(label, category="compute") if event_log else contextlib.nullcontext()


def _shape(stage, batch: torch.Tensor) -> tuple[int, int, int]:
    return stage.activation_shape(batch.size(0), batch.size(1), stage.hidden_size)


def run_1f1b(
    stage, n_micro: int, batches: list, targets: list | None = None, event_log=None
) -> float:
    previous, following = p2p.send_recv_pairs(stage.stage_index, stage.n_stages)
    dtype = next(stage.parameters()).dtype
    warmup = warmup_steps(stage.stage_index, stage.n_stages, n_micro)
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    losses: dict[int, torch.Tensor] = {}
    total_loss = 0.0

    def get_input(micro: int) -> torch.Tensor:
        if stage.is_first:
            return batches[micro]
        return p2p.recv_activation(_shape(stage, batches[micro]), dtype, previous)

    def forward(micro: int, value: torch.Tensor) -> None:
        nonlocal total_loss
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

    def backward(micro: int, grad: torch.Tensor | None) -> None:
        with _span(event_log, f"B{micro}"):
            if stage.is_last:
                (losses[micro] / n_micro).backward()
            else:
                outputs[micro].backward(grad)

    def release(micro: int) -> None:
        inputs.pop(micro, None)
        outputs.pop(micro, None)
        losses.pop(micro, None)

    # Warmup runs one short. The last warmup forward is held back so that its
    # send can be paired with the first gradient receive in the steady state
    # below; issuing it here would leave the exchange unbalanced by one.
    for micro in range(max(0, warmup - 1)):
        forward(micro, get_input(micro))
        if not stage.is_last:
            p2p.send_activation(outputs[micro], following)

    current_forward = warmup - 1
    forward(current_forward, get_input(current_forward))
    current_backward = 0

    while current_forward < n_micro:
        if stage.is_last:
            grad = None
        else:
            grad = p2p.exchange(
                outputs[current_forward],
                following,
                tuple(outputs[current_backward].shape),
                outputs[current_backward].dtype,
                following,
            )
        backward(current_backward, grad)

        next_forward = current_forward + 1
        if next_forward < n_micro:
            if stage.is_first:
                value = batches[next_forward]
            else:
                value = p2p.exchange(
                    inputs[current_backward].grad,
                    previous,
                    _shape(stage, batches[next_forward]),
                    dtype,
                    previous,
                )
                value.requires_grad_(True)
            release(current_backward)
            forward(next_forward, value)
            current_forward = next_forward
            current_backward += 1
            continue

        if not stage.is_first:
            p2p.send_activation(inputs[current_backward].grad, previous)
        release(current_backward)
        current_backward += 1
        break

    for micro in range(current_backward, n_micro):
        if stage.is_last:
            grad = None
        else:
            grad = p2p.recv_activation(
                tuple(outputs[micro].shape),
                outputs[micro].dtype,
                following,
                requires_grad=False,
            )
        backward(micro, grad)
        if not stage.is_first:
            p2p.send_activation(inputs[micro].grad, previous)
        release(micro)

    return total_loss
