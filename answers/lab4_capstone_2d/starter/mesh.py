from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

__all__ = ["MeshLayout", "build_mesh", "verify_against_device_mesh"]


@dataclass
class MeshLayout:
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
        rank = self.dp_rank * self.tp_size + self.tp_rank
        return (
            f"rank {rank} of {self.world_size}: "
            f"dp={self.dp_rank}/{self.dp_size} tp={self.tp_rank}/{self.tp_size}  "
            f"dp_group={list(self.dp_ranks)} tp_group={list(self.tp_ranks)}"
        )


def build_mesh(dp_size: int, tp_size: int) -> MeshLayout:
    rank = dist.get_rank()
    world = dist.get_world_size()
    requested = dp_size * tp_size
    if requested != world:
        raise ValueError(
            f"dp_size {dp_size} * tp_size {tp_size} = {requested}, world size is {world}"
        )

    tp_rank = rank % tp_size
    dp_rank = rank // tp_size
    my_tp_group = my_dp_group = None
    my_tp_ranks: tuple[int, ...] = ()
    my_dp_ranks: tuple[int, ...] = ()

    # new_group is collective even for ranks outside the group
    for row in range(dp_size):
        ranks = tuple(range(row * tp_size, (row + 1) * tp_size))
        group = dist.new_group(list(ranks))
        if rank in ranks:
            my_tp_group, my_tp_ranks = group, ranks

    for column in range(tp_size):
        ranks = tuple(row * tp_size + column for row in range(dp_size))
        group = dist.new_group(list(ranks))
        if rank in ranks:
            my_dp_group, my_dp_ranks = group, ranks

    return MeshLayout(
        dp_size,
        tp_size,
        dp_rank,
        tp_rank,
        my_dp_group,
        my_tp_group,
        my_dp_ranks,
        my_tp_ranks,
    )


def verify_against_device_mesh(layout: MeshLayout) -> bool:
    try:
        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh(
            "cpu",
            (layout.dp_size, layout.tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        coordinate = tuple(device_mesh.get_coordinate())
        tp_ranks = tuple(int(rank) for rank in device_mesh.mesh[layout.dp_rank, :].tolist())
        dp_ranks = tuple(int(rank) for rank in device_mesh.mesh[:, layout.tp_rank].tolist())
        return (
            coordinate == (layout.dp_rank, layout.tp_rank)
            and tp_ranks == layout.tp_ranks
            and dp_ranks == layout.dp_ranks
        )
    except Exception:
        return False
