from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from src.core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
from src.engine import evaluate_losses
from src.metrics import compute_ap, mean_average_precision
from src.utils import CLASS_TO_IDX, VOC_CLASSES, project_path, utc_timestamp


@dataclass(frozen=True)
class EvaluationConfig:
    model: str
    train_data_root: str
    test_data_root: str
    checkpoint: str
    image_size: int
    batch_size: int
    workers: int


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Assignment 2 detector.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--train-data-root", type=str, default="data/train-validation-data")
    parser.add_argument("--test-data-root", type=str, default="data/test-data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def _extract_predictions(output):
    if isinstance(output, list):
        return output
    if hasattr(output, "predictions") and output.predictions is not None:
        return output.predictions
    raise TypeError("Unsupported model output type for prediction extraction.")


def _collect_predictions_and_targets(model, loader, device: torch.device, desc: str):
    model.eval()
    predictions = []
    targets = []
    progress = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    with torch.inference_mode():
        for step, (images, batch_targets, _meta) in enumerate(progress, start=1):
            images_device = [image.to(device) for image in images]
            targets_device = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
                for target in batch_targets
            ]
            output = model(images_device, targets_device)
            batch_predictions = _extract_predictions(output)

            for pred in batch_predictions:
                predictions.append({k: v.detach().cpu() for k, v in pred.items()})
            for target in batch_targets:
                targets.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in target.items()})
            progress.set_postfix({"images": str(len(predictions)), "batches": str(step)})
    return predictions, targets


def _pr_curve_for_class(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_idx: int,
    iou_threshold: float = 0.5,
) -> dict[str, float | int | list[float]]:
    class_predictions = []
    gt_by_image = defaultdict(list)
    gt_count = 0

    for image_idx, target in enumerate(targets):
        gt_mask = target["labels"] == class_idx
        gt_boxes = target["boxes"][gt_mask]
        gt_difficult = target["difficult"][gt_mask]
        for box, diff in zip(gt_boxes, gt_difficult):
            difficult = bool(diff.item())
            gt_by_image[image_idx].append({"box": box, "matched": False, "difficult": difficult})
            if not difficult:
                gt_count += 1

    for image_idx, pred in enumerate(predictions):
        pred_mask = pred["labels"] == class_idx
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores = pred["scores"][pred_mask]
        for box, score in zip(pred_boxes, pred_scores):
            class_predictions.append({"image_idx": image_idx, "box": box, "score": float(score.item())})

    class_predictions.sort(key=lambda item: item["score"], reverse=True)
    if gt_count == 0:
        return {
            "ap": 0.0,
            "recall": [0.0, 1.0],
            "precision": [0.0, 0.0],
            "gt_count": 0,
            "prediction_count": len(class_predictions),
        }

    tp = torch.zeros(len(class_predictions))
    fp = torch.zeros(len(class_predictions))

    for pred_idx, prediction in enumerate(class_predictions):
        candidates = gt_by_image[prediction["image_idx"]]
        if not candidates:
            fp[pred_idx] = 1
            continue

        candidate_boxes = torch.stack([entry["box"] for entry in candidates], dim=0)
        ious = box_iou(prediction["box"].unsqueeze(0), candidate_boxes).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)

        if best_iou >= iou_threshold:
            entry = candidates[int(best_idx.item())]
            if entry["difficult"]:
                continue
            if not entry["matched"]:
                tp[pred_idx] = 1
                entry["matched"] = True
            else:
                fp[pred_idx] = 1
        else:
            fp[pred_idx] = 1

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(gt_count, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)
    ap = compute_ap(recall, precision)
    return {
        "ap": float(ap),
        "recall": [float(value) for value in recall.tolist()],
        "precision": [float(value) for value in precision.tolist()],
        "gt_count": int(gt_count),
        "prediction_count": len(class_predictions),
    }


def _best_iou_per_gt(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_idx: int,
) -> list[float]:
    values: list[float] = []
    for pred, target in zip(predictions, targets):
        gt_boxes = target["boxes"][target["labels"] == class_idx]
        pred_boxes = pred["boxes"][pred["labels"] == class_idx]
        if gt_boxes.numel() == 0:
            continue
        if pred_boxes.numel() == 0:
            values.extend([0.0] * gt_boxes.shape[0])
            continue
        ious = box_iou(gt_boxes, pred_boxes)
        values.extend([float(v) for v in ious.max(dim=1).values.tolist()])
    return values


def _build_plot_artifacts(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict:
    pr_curves = {}
    iou_distribution = {}
    for class_name in VOC_CLASSES:
        class_idx = CLASS_TO_IDX[class_name]
        pr_curves[class_name] = _pr_curve_for_class(predictions, targets, class_idx=class_idx, iou_threshold=0.5)
        iou_distribution[class_name] = _best_iou_per_gt(predictions, targets, class_idx=class_idx)
    return {
        "classes": VOC_CLASSES,
        "pr_curves": pr_curves,
        "iou_distribution": iou_distribution,
        "image_count": len(targets),
    }


class EvaluationApp:
    def __init__(self, config: EvaluationConfig) -> None:
        self.runtime = RuntimeEnvironment()
        checkpoint = ModelFactory.load_checkpoint_payload(config.checkpoint)
        resolved_image_size = ModelFactory.resolve_image_size(checkpoint, config.image_size)
        self.config = EvaluationConfig(
            model=config.model,
            train_data_root=config.train_data_root,
            test_data_root=config.test_data_root,
            checkpoint=config.checkpoint,
            image_size=resolved_image_size,
            batch_size=config.batch_size,
            workers=config.workers,
        )
        self.model, _checkpoint, _resolved_image_size = ModelFactory.load_checkpoint(
            config.model,
            checkpoint_path=config.checkpoint,
            image_size=resolved_image_size,
            device=self.runtime.device,
        )
        self.audit = AuditLogger(Path(config.checkpoint).parent)
        self.data = DataPipelineManager(
            DataConfig(
                train_data_root=self.config.train_data_root,
                test_data_root=self.config.test_data_root,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                workers=self.config.workers,
                subset_size=None,
                seed=42,
            )
        )

    def run(self) -> None:
        test_dataset = self.data.test_dataset()
        test_loader = self.data.test_loader()
        losses = evaluate_losses(self.model, test_loader, self.runtime.device, desc=f"{self.config.model} loss")
        predictions, targets = _collect_predictions_and_targets(
            self.model,
            test_loader,
            self.runtime.device,
            desc=f"{self.config.model} eval",
        )
        metrics = mean_average_precision(predictions=predictions, targets=targets)
        plot_data = _build_plot_artifacts(predictions, targets)

        self.audit.write_runtime("evaluation_run_manifest.json", self.config, self.model, self.runtime.device)
        self.audit.write_dataset(
            "evaluation_dataset_manifest.json",
            test_dataset=test_dataset,
            test_root=self.config.test_data_root,
            subset_size=None,
            seed=42,
        )
        self.audit.write_record(
            "evaluation_report.json",
            {
                "timestamp_utc": utc_timestamp(),
                "checkpoint_path": project_path(self.config.checkpoint),
                "losses": losses,
                "metrics": metrics,
            },
        )
        self.audit.write_record(
            "evaluation_plot_data.json",
            {
                "timestamp_utc": utc_timestamp(),
                "checkpoint_path": project_path(self.config.checkpoint),
                "model": self.config.model,
                "image_size": self.config.image_size,
                "metrics": metrics,
                "plot_data": plot_data,
            },
        )
        print({"losses": losses, "metrics": metrics})


def main():
    args = parse_args()
    config = EvaluationConfig(
        model=args.model,
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    EvaluationApp(config).run()


if __name__ == "__main__":
    main()
