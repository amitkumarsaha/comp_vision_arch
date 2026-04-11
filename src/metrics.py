from __future__ import annotations

from collections import defaultdict

import torch
from torchvision.ops import box_iou

from .utils import VOC_CLASSES


def compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for idx in range(precisions.numel() - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])

    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap.item())


def mean_average_precision(predictions, targets, num_classes: int = len(VOC_CLASSES), iou_threshold: float = 0.5):
    aps = {}

    for cls_id in range(1, num_classes + 1):
        cls_predictions = []
        gt_by_image = defaultdict(list)
        gt_count = 0

        for image_idx, target in enumerate(targets):
            mask = target["labels"] == cls_id
            boxes = target["boxes"][mask]
            difficult = target.get("difficult", torch.zeros(len(target["labels"]), dtype=torch.long))[mask]
            for box, diff in zip(boxes, difficult):
                gt_by_image[image_idx].append({"box": box, "matched": False, "difficult": bool(diff.item())})
                if not bool(diff.item()):
                    gt_count += 1

        for image_idx, pred in enumerate(predictions):
            mask = pred["labels"] == cls_id
            boxes = pred["boxes"][mask]
            scores = pred["scores"][mask]
            for box, score in zip(boxes, scores):
                cls_predictions.append({"image_idx": image_idx, "box": box, "score": float(score.item())})

        cls_predictions.sort(key=lambda item: item["score"], reverse=True)
        if gt_count == 0:
            aps[cls_id] = 0.0
            continue

        true_positive = torch.zeros(len(cls_predictions))
        false_positive = torch.zeros(len(cls_predictions))

        for pred_idx, prediction in enumerate(cls_predictions):
            entries = gt_by_image[prediction["image_idx"]]
            if not entries:
                false_positive[pred_idx] = 1
                continue

            gt_boxes = torch.stack([entry["box"] for entry in entries], dim=0)
            ious = box_iou(prediction["box"].unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)

            if best_iou >= iou_threshold:
                entry = entries[int(best_idx.item())]
                if entry["difficult"]:
                    continue
                if not entry["matched"]:
                    true_positive[pred_idx] = 1
                    entry["matched"] = True
                else:
                    false_positive[pred_idx] = 1
            else:
                false_positive[pred_idx] = 1

        tp_cum = torch.cumsum(true_positive, dim=0)
        fp_cum = torch.cumsum(false_positive, dim=0)
        recalls = tp_cum / max(gt_count, 1)
        precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)
        aps[cls_id] = compute_ap(recalls, precisions)

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    summary = {VOC_CLASSES[cls_id - 1]: score for cls_id, score in aps.items()}
    summary["mAP@0.5"] = mean_ap
    return summary
