from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, nms
from transformers import Dinov2Model

from ..utils import VOC_CLASSES

LEGACY_STEM_WEIGHT_KEY = "head.stem.0.weight"
LEGACY_CHECKPOINT_VERSION = "legacy_grid_head_v1"
CURRENT_CHECKPOINT_VERSION = "grid_head_v2"


@dataclass
class DinoDetectorOutput:
    losses: dict[str, torch.Tensor] | None
    predictions: list[dict[str, torch.Tensor]] | None


class FrozenDinov2Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name, attn_implementation="eager")
        self.hidden_size = self.model.config.hidden_size
        image_mean = getattr(self.model.config, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.model.config, "image_std", [0.229, 0.224, 0.225])
        mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.image_mean) / self.image_std
        with torch.no_grad():
            outputs = self.model(pixel_values=normalized)
        tokens = outputs.last_hidden_state[:, 1:, :]
        batch_size, num_tokens, channels = tokens.shape
        side = int(num_tokens**0.5)
        return tokens.transpose(1, 2).reshape(batch_size, channels, side, side)


class GridDetectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.cls_tower = self._make_tower(hidden_dim)
        self.box_tower = self._make_tower(hidden_dim)
        self.obj_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    @staticmethod
    def _make_tower(hidden_dim: int) -> nn.Sequential:
        layers = []
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                    nn.GELU(),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.projection(features)
        cls_features = self.cls_tower(x)
        box_features = self.box_tower(x)
        obj_logits = self.obj_head(cls_features)
        cls_logits = self.cls_head(cls_features)
        box_pred = torch.sigmoid(self.box_head(box_features))
        return obj_logits, cls_logits, box_pred


class LegacyGridDetectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(hidden_dim, num_classes + 1, kernel_size=1)
        self.box_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(features)
        return self.cls_head(x), torch.sigmoid(self.box_head(x))


class BaseDinoGridDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        hidden_dim: int = 256,
        num_classes: int = len(VOC_CLASSES),
        image_size: int = 448,
    ) -> None:
        super().__init__()
        self.backbone = FrozenDinov2Backbone(model_name=model_name)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.image_size = image_size

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _empty_prediction(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
        }

    def _boxes_to_image_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        xyxy = torch.zeros_like(boxes)
        cx = boxes[:, 0] * self.image_size
        cy = boxes[:, 1] * self.image_size
        w = boxes[:, 2] * self.image_size
        h = boxes[:, 3] * self.image_size
        xyxy[:, 0] = (cx - w / 2).clamp(0, self.image_size - 1)
        xyxy[:, 1] = (cy - h / 2).clamp(0, self.image_size - 1)
        xyxy[:, 2] = (cx + w / 2).clamp(0, self.image_size - 1)
        xyxy[:, 3] = (cy + h / 2).clamp(0, self.image_size - 1)
        return xyxy

    def _decode_selected_predictions(
        self,
        box_pred: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        score_threshold: float,
    ) -> list[dict[str, torch.Tensor]]:
        predictions = []
        for batch_idx in range(scores.shape[0]):
            keep = scores[batch_idx] >= score_threshold
            if keep.sum() == 0:
                predictions.append(self._empty_prediction(box_pred.device))
                continue

            selected_boxes = box_pred[batch_idx, :, keep].transpose(0, 1)
            selected_scores = scores[batch_idx][keep]
            selected_labels = labels[batch_idx][keep]
            xyxy = self._boxes_to_image_xyxy(selected_boxes)
            keep_indices = nms(xyxy, selected_scores, iou_threshold=0.5)
            predictions.append(
                {
                    "boxes": xyxy[keep_indices],
                    "scores": selected_scores[keep_indices],
                    "labels": selected_labels[keep_indices],
                }
            )
        return predictions


class DinoGridDetector(BaseDinoGridDetector):
    checkpoint_version = CURRENT_CHECKPOINT_VERSION

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        hidden_dim: int = 256,
        num_classes: int = len(VOC_CLASSES),
        image_size: int = 448,
    ) -> None:
        super().__init__(model_name=model_name, hidden_dim=hidden_dim, num_classes=num_classes, image_size=image_size)
        self.head = GridDetectionHead(self.backbone.hidden_size, hidden_dim, num_classes)

    def encode_targets(self, targets, grid_h: int, grid_w: int, device: torch.device):
        cls_target = torch.full((len(targets), grid_h, grid_w), -1, dtype=torch.long, device=device)
        box_target = torch.zeros((len(targets), 4, grid_h, grid_w), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((len(targets), grid_h, grid_w), dtype=torch.bool, device=device)

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue

            centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / self.image_size).clamp(0, 0.9999)
            centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / self.image_size).clamp(0, 0.9999)
            widths = ((boxes[:, 2] - boxes[:, 0]) / self.image_size).clamp(0, 1)
            heights = ((boxes[:, 3] - boxes[:, 1]) / self.image_size).clamp(0, 1)
            areas = widths * heights
            order = torch.argsort(areas, descending=True)

            cell_x = torch.floor(centers_x * grid_w).long()
            cell_y = torch.floor(centers_y * grid_h).long()

            for obj_idx in order.tolist():
                gx = cell_x[obj_idx].item()
                gy = cell_y[obj_idx].item()
                if pos_mask[batch_idx, gy, gx]:
                    continue
                cls_target[batch_idx, gy, gx] = labels[obj_idx] - 1
                box_target[batch_idx, :, gy, gx] = torch.tensor(
                    [centers_x[obj_idx], centers_y[obj_idx], widths[obj_idx], heights[obj_idx]],
                    dtype=torch.float32,
                    device=device,
                )
                pos_mask[batch_idx, gy, gx] = True
        return cls_target, box_target, pos_mask

    def compute_losses(self, obj_logits: torch.Tensor, cls_logits: torch.Tensor, box_pred: torch.Tensor, targets):
        device = cls_logits.device
        _, _, grid_h, grid_w = cls_logits.shape
        cls_target, box_target, pos_mask = self.encode_targets(targets, grid_h, grid_w, device)

        obj_target = pos_mask.unsqueeze(1).float()
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

        if pos_mask.any():
            positive_cls_logits = cls_logits.permute(0, 2, 3, 1)[pos_mask]
            positive_cls_target = cls_target[pos_mask]
            cls_loss = F.cross_entropy(positive_cls_logits, positive_cls_target)
        else:
            cls_loss = cls_logits.sum() * 0.0

        positive = pos_mask.unsqueeze(1).expand(-1, 4, -1, -1)
        if positive.any():
            pred_boxes = box_pred[positive].view(-1, 4)
            true_boxes = box_target[positive].view(-1, 4)
            l1_loss = F.l1_loss(pred_boxes, true_boxes)
            giou = generalized_box_iou(self.cxcywh_to_xyxy(pred_boxes), self.cxcywh_to_xyxy(true_boxes))
            giou_loss = 1.0 - torch.diag(giou).mean()
        else:
            l1_loss = obj_logits.sum() * 0.0
            giou_loss = obj_logits.sum() * 0.0

        total = 2.0 * obj_loss + cls_loss + 5.0 * l1_loss + 2.0 * giou_loss
        return {
            "loss": total,
            "loss_obj": obj_loss,
            "loss_cls": cls_loss,
            "loss_l1": l1_loss,
            "loss_giou": giou_loss,
        }

    def decode_predictions(
        self,
        obj_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        box_pred: torch.Tensor,
        score_threshold: float = 0.20,
    ):
        objectness = torch.sigmoid(obj_logits).squeeze(1)
        class_probabilities = torch.softmax(cls_logits, dim=1)
        class_scores, labels = class_probabilities.max(dim=1)
        scores = objectness * class_scores
        labels = labels + 1
        return self._decode_selected_predictions(box_pred, scores, labels, score_threshold)

    def forward(self, images, targets=None):
        image_tensor = torch.stack(images, dim=0)
        features = self.backbone(image_tensor)
        obj_logits, cls_logits, box_pred = self.head(features)
        losses = self.compute_losses(obj_logits, cls_logits, box_pred, targets) if targets is not None else None
        predictions = self.decode_predictions(obj_logits, cls_logits, box_pred) if (not self.training or targets is None) else None
        return DinoDetectorOutput(losses=losses, predictions=predictions)


class LegacyDinoGridDetector(BaseDinoGridDetector):
    checkpoint_version = LEGACY_CHECKPOINT_VERSION

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        hidden_dim: int = 256,
        num_classes: int = len(VOC_CLASSES),
        image_size: int = 448,
    ) -> None:
        super().__init__(model_name=model_name, hidden_dim=hidden_dim, num_classes=num_classes, image_size=image_size)
        self.head = LegacyGridDetectionHead(self.backbone.hidden_size, hidden_dim, num_classes)

    def encode_targets(self, targets, grid_h: int, grid_w: int, device: torch.device):
        cls_target = torch.zeros((len(targets), grid_h, grid_w), dtype=torch.long, device=device)
        box_target = torch.zeros((len(targets), 4, grid_h, grid_w), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((len(targets), grid_h, grid_w), dtype=torch.bool, device=device)

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue

            centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / self.image_size).clamp(0, 0.9999)
            centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / self.image_size).clamp(0, 0.9999)
            widths = ((boxes[:, 2] - boxes[:, 0]) / self.image_size).clamp(0, 1)
            heights = ((boxes[:, 3] - boxes[:, 1]) / self.image_size).clamp(0, 1)
            areas = widths * heights
            order = torch.argsort(areas, descending=True)

            cell_x = torch.floor(centers_x * grid_w).long()
            cell_y = torch.floor(centers_y * grid_h).long()

            for obj_idx in order.tolist():
                gx = cell_x[obj_idx].item()
                gy = cell_y[obj_idx].item()
                if pos_mask[batch_idx, gy, gx]:
                    continue
                cls_target[batch_idx, gy, gx] = labels[obj_idx]
                box_target[batch_idx, :, gy, gx] = torch.tensor(
                    [centers_x[obj_idx], centers_y[obj_idx], widths[obj_idx], heights[obj_idx]],
                    dtype=torch.float32,
                    device=device,
                )
                pos_mask[batch_idx, gy, gx] = True
        return cls_target, box_target, pos_mask

    def compute_losses(self, cls_logits: torch.Tensor, box_pred: torch.Tensor, targets):
        device = cls_logits.device
        _, _, grid_h, grid_w = cls_logits.shape
        cls_target, box_target, pos_mask = self.encode_targets(targets, grid_h, grid_w, device)

        cls_loss = F.cross_entropy(cls_logits, cls_target)
        positive = pos_mask.unsqueeze(1).expand(-1, 4, -1, -1)
        if positive.any():
            pred_boxes = box_pred[positive].view(-1, 4)
            true_boxes = box_target[positive].view(-1, 4)
            l1_loss = F.l1_loss(pred_boxes, true_boxes)
            giou = generalized_box_iou(self.cxcywh_to_xyxy(pred_boxes), self.cxcywh_to_xyxy(true_boxes))
            giou_loss = 1.0 - torch.diag(giou).mean()
        else:
            l1_loss = cls_logits.sum() * 0.0
            giou_loss = cls_logits.sum() * 0.0

        total = cls_loss + 5.0 * l1_loss + 2.0 * giou_loss
        return {"loss": total, "loss_cls": cls_loss, "loss_l1": l1_loss, "loss_giou": giou_loss}

    def decode_predictions(self, cls_logits: torch.Tensor, box_pred: torch.Tensor, score_threshold: float = 0.25):
        probabilities = torch.softmax(cls_logits, dim=1)
        scores, labels = probabilities[:, 1:, :, :].max(dim=1)
        labels = labels + 1
        return self._decode_selected_predictions(box_pred, scores, labels, score_threshold)

    def forward(self, images, targets=None):
        image_tensor = torch.stack(images, dim=0)
        features = self.backbone(image_tensor)
        cls_logits, box_pred = self.head(features)
        losses = self.compute_losses(cls_logits, box_pred, targets) if targets is not None else None
        predictions = self.decode_predictions(cls_logits, box_pred) if (not self.training or targets is None) else None
        return DinoDetectorOutput(losses=losses, predictions=predictions)


class DinoCheckpointCompatibility:
    @staticmethod
    def version_from_checkpoint(checkpoint: dict) -> str:
        metadata_version = checkpoint.get("head_version")
        if metadata_version:
            return str(metadata_version)
        state_dict = checkpoint["state_dict"]
        if LEGACY_STEM_WEIGHT_KEY in state_dict:
            return LEGACY_CHECKPOINT_VERSION
        return CURRENT_CHECKPOINT_VERSION

    @classmethod
    def build_model(
        cls,
        checkpoint: dict,
        image_size: int = 448,
        model_name: str = "facebook/dinov2-small",
    ):
        version = cls.version_from_checkpoint(checkpoint)
        if version == LEGACY_CHECKPOINT_VERSION:
            model = LegacyDinoGridDetector(model_name=model_name, image_size=image_size)
        else:
            model = DinoGridDetector(model_name=model_name, image_size=image_size)
        model.load_state_dict(checkpoint["state_dict"])
        return model


def is_legacy_dino_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    return LEGACY_STEM_WEIGHT_KEY in state_dict


def build_dino_model_for_checkpoint(
    checkpoint: dict,
    image_size: int = 448,
    model_name: str = "facebook/dinov2-small",
):
    return DinoCheckpointCompatibility.build_model(
        checkpoint=checkpoint,
        image_size=image_size,
        model_name=model_name,
    )
