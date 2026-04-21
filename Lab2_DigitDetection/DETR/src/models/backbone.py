"""ResNet backbone for DETR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from src.utils.misc import NestedTensor


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone with optional DC5 dilation."""

    def __init__(self, pretrained: bool = True, dilation: bool = True) -> None:
        super().__init__()

        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            except AttributeError:
                weights = "IMAGENET1K_V2"

        backbone = torchvision.models.resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, dilation],
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.num_channels = 2048

    def forward(self, tensor_list: NestedTensor) -> NestedTensor:
        x, mask = tensor_list.decompose()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        resized_mask = F.interpolate(
            mask[None].float(),
            size=x.shape[-2:],
            mode="nearest",
        )[0].to(dtype=torch.bool)
        return NestedTensor(x, resized_mask)
