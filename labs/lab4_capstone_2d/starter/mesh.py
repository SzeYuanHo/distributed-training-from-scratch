"""Step 1: build the 2D mesh and both process groups by hand.

``world_size = dp * tp``. Every rank belongs to exactly one DP group and exactly
one TP group, and which ranks share which group decides which traffic crosses
which link.

⚠ **Tensor-parallel groups must be the innermost, fastest dimension.**

On an 8-GPU NVLink node with ``dp=2, tp=4``, you want ranks 0-3 forming one TP
group and 4-7 the other, so every TP all-reduce stays on NVLink. Assign it the
other way and your TP groups span the slow inter-node link. Both versions run.
One is three times slower.

Lab 2's ceiling derivation is why: TP communicates the most and tolerates latency
the least, so it gets the best link. DP communicates once per step and overlaps
with the backward pass, so it can live on the worst one.

You build the groups by hand here and then cross-check against
``torch.distributed.device_mesh.init_device_mesh``. The test asserts the
coordinate mapping, because "it ran without crashing" is not evidence you got the
layout right.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MeshLayout", "build_mesh", "verify_against_device_mesh"]


@dataclass
class MeshLayout:
    """This rank's position in the mesh, plus both process groups.

    Attributes:
        dp_size, tp_size: mesh dimensions.
        dp_rank, tp_rank: this rank's coordinates.
        dp_group, tp_group: the process groups to pass to collectives.
        dp_ranks, tp_ranks: the global rank ids in each of this rank's groups,
            which is what the test inspects and what you print for the writeup.
    """

    dp_size: int
    tp_size: int
    dp_rank: int
    tp_rank: int
    dp_group: object = None
    tp_group: object = None
    dp_ranks: tuple[int, ...] = ()
    tp_ranks: tuple[int, ...] = ()

    @property
    def world_size(self) -> int:
        return self.dp_size * self.tp_size

    def __str__(self) -> str:
        return (
            f"rank {self.dp_rank * self.tp_size + self.tp_rank} of {self.world_size}: "
            f"dp={self.dp_rank}/{self.dp_size} tp={self.tp_rank}/{self.tp_size}  "
            f"dp_group={list(self.dp_ranks)} tp_group={list(self.tp_ranks)}"
        )


def build_mesh(dp_size: int, tp_size: int) -> MeshLayout:
    """Construct both process groups for this rank's position in a ``dp x tp`` mesh.

    Args:
        dp_size: data-parallel degree.
        tp_size: tensor-parallel degree. The **innermost** dimension.

    Returns:
        This rank's :class:`MeshLayout`.

    **The rank arithmetic**, with TP innermost:

    ```
    global_rank = dp_rank * tp_size + tp_rank
    tp_rank     = global_rank % tp_size
    dp_rank     = global_rank // tp_size
    ```

    So at ``dp=2, tp=4``: TP groups are ``[0,1,2,3]`` and ``[4,5,6,7]``, and DP
    groups are ``[0,4]``, ``[1,5]``, ``[2,6]``, ``[3,7]``. Contiguous ranks share a
    TP group, which on a real machine means contiguous ranks share a node.

    ⚠ **Every rank must call ``dist.new_group`` for every group, in the same
    order.** ``new_group`` is collective: all ranks participate in creating each
    group even though only some belong to it. Building only your own groups
    deadlocks on the first call, and the hang looks like a bug in your model.

    So the loop is: for each of the ``dp_size`` TP groups, every rank calls
    ``new_group``; keep the handle only if you are in it. Then the same for the
    ``tp_size`` DP groups.

    Raises:
        ValueError: if ``dp_size * tp_size`` does not equal the world size, with
            all three numbers in the message.
    """
    raise NotImplementedError(
        "Step 1a: compute the coordinates, then create every group on every rank."
    )


def verify_against_device_mesh(layout: MeshLayout) -> bool:
    """Cross-check the hand-built mesh against ``init_device_mesh``.

    ::

        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    Torch's mesh uses the same convention: the *last* dimension varies fastest,
    so passing ``(dp, tp)`` puts TP innermost, matching what you built. Confirming
    that rather than assuming it is worth five minutes, because the convention is
    the kind of thing that flips between versions.

    Returns:
        True when the coordinates and group memberships agree. False when torch
        is unavailable or the mesh cannot be built on this backend, so the caller
        can skip rather than fail.
    """
    raise NotImplementedError("Step 1b.")
