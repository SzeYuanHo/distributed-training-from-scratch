"""Step 4: communicate each gradient the moment it exists.

``SimpleDDP`` waits for the whole backward pass, then communicates everything.
That leaves the network idle for the entire backward and then the compute idle for
the entire all-reduce. This class fixes the first half: fire a collective as soon
as each gradient is ready.

Still synchronous, so it is not yet faster. It is the scaffolding that makes
step 6's overlap possible, and it is where you decide which hook API to use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["HookedDDP"]


class HookedDDP(nn.Module):
    """DDP that all-reduces each gradient from an autograd hook.

    No explicit ``sync_gradients()`` call. By the time ``loss.backward()`` returns,
    every gradient has already been averaged.

    ⚠ **The API question, and it is the important part of this lab.**

    ``nn.Module.register_backward_hook`` is deprecated, has documented incorrect
    behaviour for several module types, and does not tell you when an individual
    parameter's gradient is ready. It is what most tutorials written before 2023
    reach for. Do not use it.

    Three that do work, with the trade-off you are choosing between:

    ``param.register_post_accumulate_grad_hook(fn)``
        Fires after ``param.grad`` has been accumulated, once per parameter per
        backward. Receives the parameter. Needs torch 2.1 or later.

    ``param.register_hook(fn)``
        Fires when the gradient *with respect to that tensor* is computed, which
        is before accumulation into ``.grad``. Receives the gradient tensor, and
        can fire more than once for a parameter used in several places, which
        this repo's tied embedding is. If you pick this one you have to handle
        the multiple firings yourself.

    ``param.expand_as(param).grad_fn.next_functions[0][0].register_hook(fn)``
        Registers on the ``AccumulateGrad`` autograd node directly. This is what
        ``reducer.cpp`` does. Ugly, and implementing it once makes the PyTorch
        source readable, which is worth something.

    The tests check behaviour, not your choice. Leave a comment saying which you
    picked and why.

    **Handles must be released.** Hooks registered on parameters keep a reference
    to ``self``, and ``self`` holds the module, so a wrapper that never removes
    its hooks leaks the whole model. Keep the handles and remove them in
    ``__del__`` or an explicit ``close()``. This matters here because the tests
    construct dozens of these.

    **Every rank must issue the same collectives in the same order.** Hooks fire in
    autograd's order, which is deterministic for a fixed graph, so this holds as
    long as every rank ran the same forward pass. A model with a data-dependent
    branch breaks it, and the symptom is a hang rather than an error. Step 8 is
    about exactly that.
    """

    def __init__(self, module: nn.Module, process_group=None):
        super().__init__()
        raise NotImplementedError(
            "Step 4a: broadcast parameters, then register one hook per parameter."
        )

    def _make_hook(self, param: nn.Parameter):
        """Build the callback for one parameter.

        Watch the closure. Writing ``for p in params: p.register_hook(lambda g: f(p))``
        captures ``p`` by reference, so every hook sees the last parameter in the
        loop. Bind it explicitly, either with a default argument or with a factory
        like this one. The bug is silent: gradients get averaged, just the wrong
        ones, and the loss curve is subtly off.
        """
        raise NotImplementedError("Step 4b: return a callable that all-reduces this grad.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Step 4c.")

    def parameters(self, recurse: bool = True):
        raise NotImplementedError("Step 4d.")

    def close(self) -> None:
        """Remove every registered hook. Call it when you are done with the wrapper."""
        raise NotImplementedError("Step 4e.")
