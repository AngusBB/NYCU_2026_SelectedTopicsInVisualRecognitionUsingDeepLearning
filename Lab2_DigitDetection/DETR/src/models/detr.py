"""DETR model, criterion, and post-processing."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from src.models.backbone import ResNetBackbone
from src.models.matcher import HungarianMatcher
from src.models.position_encoding import PositionEmbeddingSine
from src.models.transformer import Transformer
from src.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from src.utils.distributed import get_world_size, is_dist_avail_and_initialized
from src.utils.misc import NestedTensor, accuracy, inverse_sigmoid, nested_tensor_from_tensor_list


class DETR(nn.Module):
    """ResNet-50 DETR with optional iterative box refinement."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config["model"]

        self.num_classes = int(model_cfg["num_classes"])
        self.num_queries = int(model_cfg["num_queries"])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.aux_loss = bool(model_cfg["aux_loss"])
        self.with_box_refine = bool(model_cfg["with_box_refine"])

        self.backbone = ResNetBackbone(
            pretrained=bool(model_cfg["backbone_pretrained"]),
            dilation=bool(model_cfg["dilation"]),
        )
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=int(model_cfg["num_heads"]),
            num_encoder_layers=int(model_cfg["encoder_layers"]),
            num_decoder_layers=int(model_cfg["decoder_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            normalize_before=bool(model_cfg["pre_norm"]),
            return_intermediate_dec=True,
        )

        num_prediction_layers = self.transformer.decoder.num_layers
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)

        class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.with_box_refine:
            self.class_embed = nn.ModuleList(
                [nn.Linear(hidden_dim, self.num_classes + 1) for _ in range(num_prediction_layers)]
            )
            self.bbox_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_prediction_layers)]
            )
        else:
            self.class_embed = nn.ModuleList([class_embed for _ in range(num_prediction_layers)])
            self.bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_prediction_layers)])

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.uniform_(self.query_embed.weight)
        nn.init.uniform_(self.refpoint_embed.weight)

        for bbox_head in self.bbox_embed:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)
            bbox_head.layers[-1].bias.data[2:] = -2.0

    def forward(self, samples: List[torch.Tensor] | NestedTensor | torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(samples, (list, tuple)):
            samples = nested_tensor_from_tensor_list(list(samples))
        elif isinstance(samples, torch.Tensor):
            samples = nested_tensor_from_tensor_list(list(samples))

        features = self.backbone(samples)
        pos_embed = self.position_embedding(features)
        src, mask = features.decompose()
        hidden_states, _ = self.transformer(
            self.input_proj(src),
            mask,
            self.query_embed.weight,
            pos_embed,
        )

        batch_size = hidden_states.shape[1]
        outputs_class = []
        outputs_coord = []
        reference = self.refpoint_embed.weight.sigmoid().unsqueeze(0).expand(batch_size, -1, -1)

        for layer_index, layer_hidden in enumerate(hidden_states):
            class_logits = self.class_embed[layer_index](layer_hidden)
            box_delta = self.bbox_embed[layer_index](layer_hidden)
            if self.with_box_refine:
                if layer_index == 0:
                    box_delta = box_delta + inverse_sigmoid(reference)
                else:
                    box_delta = box_delta + inverse_sigmoid(outputs_coord[-1].detach())
            box_prediction = box_delta.sigmoid()
            outputs_class.append(class_logits)
            outputs_coord.append(box_prediction)

        outputs_class = torch.stack(outputs_class)
        outputs_coord = torch.stack(outputs_coord)

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return output

    @staticmethod
    def _set_aux_loss(
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {"pred_logits": class_logits, "pred_boxes": box_logits}
            for class_logits, box_logits in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """Compute DETR losses after Hungarian matching."""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.label_smoothing = label_smoothing

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
        num_boxes: float,
    ) -> Dict[str, torch.Tensor]:
        del num_boxes
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if idx[0].numel() > 0:
            target_classes_o = torch.cat(
                [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)],
                dim=0,
            )
            target_classes[idx] = target_classes_o
        else:
            target_classes_o = torch.empty(0, dtype=torch.int64, device=src_logits.device)

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            label_smoothing=self.label_smoothing,
        )
        losses = {"loss_ce": loss_ce}
        if target_classes_o.numel() > 0:
            losses["class_error"] = 100.0 - accuracy(src_logits[idx], target_classes_o)[0]
        else:
            losses["class_error"] = torch.tensor(0.0, device=src_logits.device)
        return losses

    @staticmethod
    def loss_cardinality(
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
        num_boxes: float,
    ) -> Dict[str, torch.Tensor]:
        del indices, num_boxes
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        target_lengths = torch.as_tensor([len(target["labels"]) for target in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_error = F.l1_loss(card_pred.float(), target_lengths.float())
        return {"cardinality_error": card_error}

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
        num_boxes: float,
    ) -> Dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [self._normalize_boxes(target, target_idx) for target, (_, target_idx) in zip(targets, indices)],
            dim=0,
        )

        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _normalize_boxes(self, target: Dict[str, torch.Tensor], target_index: torch.Tensor) -> torch.Tensor:
        boxes = target["boxes"][target_index]
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        height, width = target["size"].unbind()
        scale = torch.stack((width, height, width, height)).to(dtype=boxes.dtype)
        return box_xyxy_to_cxcywh(boxes / scale)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(source, batch_index) for batch_index, (source, _) in enumerate(indices)]
        )
        src_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss: str, outputs, targets, indices, num_boxes: float):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "cardinality": self.loss_cardinality,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes_tensor = torch.as_tensor(
            [num_boxes],
            dtype=torch.float32,
            device=outputs["pred_logits"].device,
        )
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes_tensor)
        num_boxes = torch.clamp(num_boxes_tensor / get_world_size(), min=1.0).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for layer_index, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    values = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    losses.update({f"{name}_{layer_index}": value for name, value in values.items()})

        return losses


class PostProcess(nn.Module):
    """Convert normalized DETR outputs to COCO-style predictions."""

    def __init__(self, category_ids: List[int]) -> None:
        super().__init__()
        self.register_buffer(
            "category_ids",
            torch.as_tensor(category_ids, dtype=torch.int64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], target_sizes: torch.Tensor):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob = F.softmax(out_logits, dim=-1)
        scores, labels = prob[..., :-1].max(dim=-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_height, img_width = target_sizes.unbind(1)
        scale = torch.stack([img_width, img_height, img_width, img_height], dim=1)
        boxes = boxes * scale[:, None, :]

        results = []
        for sample_scores, sample_labels, sample_boxes in zip(scores, labels, boxes):
            sample_xywh = sample_boxes.clone()
            sample_xywh[:, 2:] -= sample_xywh[:, :2]
            mapped_labels = self.category_ids.to(sample_labels.device)[sample_labels]
            results.append(
                {
                    "scores": sample_scores,
                    "labels": mapped_labels,
                    "boxes": sample_xywh,
                    "boxes_xyxy": sample_boxes,
                }
            )
        return results


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims, dims[1:] + [output_dim])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def build_detr(config: Dict) -> tuple[nn.Module, nn.Module, nn.Module]:
    model = DETR(config)
    matcher = HungarianMatcher(
        cost_class=float(config["loss"]["class_loss_coef"]),
        cost_bbox=float(config["loss"]["bbox_loss_coef"]),
        cost_giou=float(config["loss"]["giou_loss_coef"]),
    )

    weight_dict = {
        "loss_ce": float(config["loss"]["class_loss_coef"]),
        "loss_bbox": float(config["loss"]["bbox_loss_coef"]),
        "loss_giou": float(config["loss"]["giou_loss_coef"]),
    }
    if config["model"]["aux_loss"]:
        for layer in range(int(config["model"]["decoder_layers"]) - 1):
            weight_dict.update(
                {
                    f"loss_ce_{layer}": float(config["loss"]["class_loss_coef"]),
                    f"loss_bbox_{layer}": float(config["loss"]["bbox_loss_coef"]),
                    f"loss_giou_{layer}": float(config["loss"]["giou_loss_coef"]),
                }
            )

    criterion = SetCriterion(
        num_classes=int(config["model"]["num_classes"]),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=float(config["loss"]["eos_coef"]),
        losses=["labels", "boxes", "cardinality"],
        label_smoothing=float(config["loss"]["label_smoothing"]),
    )
    postprocessors = PostProcess(category_ids=list(config["model"]["category_ids"]))
    return model, criterion, postprocessors
