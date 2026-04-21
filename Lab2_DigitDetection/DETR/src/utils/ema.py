"""Exponential moving average for model weights."""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn


class ModelEMA:
    """Track an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            if not value.dtype.is_floating_point:
                value.copy_(model_state[key])
                continue
            value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)

