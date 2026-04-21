"""Distributed helpers."""

from __future__ import annotations

import builtins
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def setup_for_distributed(is_master: bool) -> None:
    """Disable printing on non-master processes."""
    builtin_print = builtins.print

    def print_fn(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print_fn


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def save_on_master(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def all_gather_object(data: Any) -> List[Any]:
    if not is_dist_avail_and_initialized():
        return [data]
    gathered: List[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, data)
    return gathered


def init_distributed_mode() -> Dict[str, Any]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return {
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    init_kwargs = {"backend": backend, "init_method": "env://"}
    if backend == "nccl":
        init_kwargs["device_id"] = local_rank
    dist.init_process_group(**init_kwargs)
    barrier()
    setup_for_distributed(rank == 0)
    return {
        "distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def destroy_process_group() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

