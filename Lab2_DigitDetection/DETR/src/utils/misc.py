"""General utilities shared across training and inference."""

from __future__ import annotations

import collections
import datetime
import random
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from src.utils.distributed import get_world_size, is_dist_avail_and_initialized


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class NestedTensor:
    tensors: Tensor
    mask: Tensor

    def to(self, device: torch.device) -> "NestedTensor":
        return NestedTensor(self.tensors.to(device), self.mask.to(device))

    def decompose(self) -> tuple[Tensor, Tensor]:
        return self.tensors, self.mask


def _max_by_axis(shapes: Sequence[Sequence[int]]) -> List[int]:
    maxes = list(shapes[0])
    for shape in shapes[1:]:
        for axis, value in enumerate(shape):
            maxes[axis] = max(maxes[axis], value)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    if tensor_list[0].ndim != 3:
        raise ValueError("Expected tensors in CHW format.")

    max_size = _max_by_axis([list(image.shape) for image in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    batch = tensor_list[0].new_zeros(batch_shape)
    mask = torch.ones(
        (len(tensor_list), max_size[1], max_size[2]),
        dtype=torch.bool,
        device=tensor_list[0].device,
    )

    for image, padded_image, padded_mask in zip(tensor_list, batch, mask):
        channels, height, width = image.shape
        padded_image[:channels, :height, :width].copy_(image)
        padded_mask[:height, :width] = False

    return NestedTensor(batch, mask)


def collate_fn(batch: Sequence[tuple[Tensor, Dict[str, Tensor]]]) -> tuple[List[Tensor], List[Dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def move_targets_to_device(
    targets: List[Dict[str, Tensor]],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    moved: List[Dict[str, Tensor]] = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def inverse_sigmoid(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> List[Tensor]:
    if target.numel() == 0:
        return [torch.zeros([], device=output.device) for _ in topk]
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target[None])

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k * (100.0 / target.numel()))
    return results


def reduce_dict(input_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys], dim=0)
        dist.all_reduce(values)
        values /= world_size
        return {key: value for key, value in zip(keys, values)}


class SmoothedValue:
    """Track a series of values and expose smoothed statistics."""

    def __init__(self, window_size: int = 20) -> None:
        self.deque: Deque[float] = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        return float(np.median(list(self.deque))) if self.deque else 0.0

    @property
    def avg(self) -> float:
        return float(np.mean(list(self.deque))) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0


class MetricLogger:
    """Lightweight metric logger with ETA reporting."""

    def __init__(self, delimiter: str = "  ") -> None:
        self.meters: Dict[str, SmoothedValue] = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.meters[key].update(float(value))

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(attr)

    def __str__(self) -> str:
        return self.delimiter.join(
            f"{name}: {meter.avg:.4f} ({meter.global_avg:.4f})"
            for name, meter in self.meters.items()
        )

    def synchronize_between_processes(self) -> None:
        if not is_dist_avail_and_initialized():
            return
        for meter in self.meters.values():
            values = torch.tensor(
                [meter.count, meter.total],
                dtype=torch.float64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            dist.barrier()
            dist.all_reduce(values)
            meter.count = int(values[0].item())
            meter.total = float(values[1].item())

    def log_every(
        self,
        iterable: Iterable,
        print_freq: int,
        header: str = "",
    ) -> Iterator:
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(window_size=10)
        data_time = SmoothedValue(window_size=10)
        size = len(iterable)
        digits = len(str(size))

        for index, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == size - 1:
                eta_seconds = iter_time.global_avg * (size - index - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"{header}[{index:{digits}d}/{size}]  "
                    f"eta: {eta_string}  {self}  "
                    f"time: {iter_time.avg:.4f}  data: {data_time.avg:.4f}"
                )
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header}Total time: {total_time_str}")

