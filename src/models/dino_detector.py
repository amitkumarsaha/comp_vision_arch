from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, nms
from transformers import Dinov2Model

from ..utils import VOC_CLASSES


@dataclass
class DinoDetectorOutput:
    losses: dict[str, torch.Tensor] | None
    predictions: list[dict[str, torch.Tensor]] | None


class FrozenDinov2Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
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


class DinoGridDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        hidden_dim: int = 256,
        num_classes: int = len(VOC_CLASSES),
        image_size: int = 448,
    ) -> None:
        super().__init__()
        self.backbone = FrozenDinov2Backbone(model_name=model_name)
        self.head = GridDetectionHead(self.backbone.hidden_size, hidden_dim, num_classes)
        self.num_classes = num_classes
        self.image_size = image_size

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

    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)

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
        predictions = []

        for batch_idx in range(cls_logits.shape[0]):
            keep = scores[batch_idx] >= score_threshold
            if keep.sum() == 0:
                predictions.append(
                    {
                        "boxes": torch.zeros((0, 4), device=cls_logits.device),
                        "scores": torch.zeros((0,), device=cls_logits.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=cls_logits.device),
                    }
                )
                continue

            selected_boxes = box_pred[batch_idx, :, keep].transpose(0, 1)
            selected_scores = scores[batch_idx][keep]
            selected_labels = labels[batch_idx][keep]

            xyxy = torch.zeros_like(selected_boxes)
            cx = selected_boxes[:, 0] * self.image_size
            cy = selected_boxes[:, 1] * self.image_size
            w = selected_boxes[:, 2] * self.image_size
            h = selected_boxes[:, 3] * self.image_size
            xyxy[:, 0] = (cx - w / 2).clamp(0, self.image_size - 1)
            xyxy[:, 1] = (cy - h / 2).clamp(0, self.image_size - 1)
            xyxy[:, 2] = (cx + w / 2).clamp(0, self.image_size - 1)
            xyxy[:, 3] = (cy + h / 2).clamp(0, self.image_size - 1)

            keep_indices = nms(xyxy, selected_scores, iou_threshold=0.5)
            predictions.append(
                {
                    "boxes": xyxy[keep_indices],
                    "scores": selected_scores[keep_indices],
                    "labels": selected_labels[keep_indices],
                }
            )
        return predictions

    def forward(self, images, targets=None):
        image_tensor = torch.stack(images, dim=0)
        features = self.backbone(image_tensor)
        cls_logits, box_pred = self.head(features)
        losses = self.compute_losses(cls_logits, box_pred, targets) if targets is not None else None
        predictions = self.decode_predictions(cls_logits, box_pred) if (not self.training or targets is None) else None
        return DinoDetectorOutput(losses=losses, predictions=predictions)
