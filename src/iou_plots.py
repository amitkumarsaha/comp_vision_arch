from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.utils import CLASS_TO_IDX, VOC_CLASSES
else:
    from .utils import CLASS_TO_IDX, VOC_CLASSES


@dataclass(frozen=True)
class IoUPlotConfig:
    test_voc_root: str
    dino_checkpoint: str
    fasterrcnn_checkpoint: str
    image_size: int | None
    batch_size: int
    workers: int
    output_dir: str
    hf_cache_dir: str


class VOCIoUTestDataset(Dataset):
    def __init__(self, voc2007_root: str | Path, image_size: int) -> None:
        self.voc_root = Path(voc2007_root)
        self.image_size = int(image_size)
        split_file = self.voc_root / "ImageSets" / "Main" / "test.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing VOC test split file: {split_file}")
        self.image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_ids = [image_id for image_id in self.image_ids if self._has_target_class(image_id)]

    def _annotation_path(self, image_id: str) -> Path:
        return self.voc_root / "Annotations" / f"{image_id}.xml"

    def _image_path(self, image_id: str) -> Path:
        return self.voc_root / "JPEGImages" / f"{image_id}.jpg"

    def _parse_objects(self, image_id: str):
        annotation_path = self._annotation_path(image_id)
        root = ET.parse(annotation_path).getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if name not in CLASS_TO_IDX:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0")) - 1.0
            ymin = float(bnd.findtext("ymin", "0")) - 1.0
            xmax = float(bnd.findtext("xmax", "0")) - 1.0
            ymax = float(bnd.findtext("ymax", "0")) - 1.0
            if xmax <= xmin or ymax <= ymin:
                continue
            difficult = int(obj.findtext("difficult", "0"))
            objects.append((name, [xmin, ymin, xmax, ymax], difficult))
        return objects

    def _has_target_class(self, image_id: str) -> bool:
        return len(self._parse_objects(image_id)) > 0

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self._image_path(image_id)).convert("RGB")
        orig_w, orig_h = image.size

        objects = self._parse_objects(image_id)
        boxes = []
        labels = []
        difficult = []
        for name, box, diff in objects:
            boxes.append(box)
            labels.append(CLASS_TO_IDX[name])
            difficult.append(diff)

        image = F.resize(image, [self.image_size, self.image_size])
        image_tensor = F.to_tensor(image)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        boxes_tensor[:, [0, 2]] *= self.image_size / orig_w
        boxes_tensor[:, [1, 3]] *= self.image_size / orig_h

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficult": torch.tensor(difficult, dtype=torch.long),
            "image_id": f"{image_id}.jpg",
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        return image_tensor, target


def collate_iou_batch(batch):
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return images, targets


class IoUPlotGenerator:
    def __init__(self, config: IoUPlotConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(config.hf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

        self.model_factory, runtime_environment = self._import_runtime_classes()
        self.runtime = runtime_environment()

    @staticmethod
    def _import_runtime_classes():
        if __package__ in (None, ""):
            from src.core.runtime import ModelFactory, RuntimeEnvironment
        else:
            from .core.runtime import ModelFactory, RuntimeEnvironment
        return ModelFactory, RuntimeEnvironment

    @staticmethod
    def _extract_predictions(output):
        if isinstance(output, list):
            return output
        if hasattr(output, "predictions") and output.predictions is not None:
            return output.predictions
        raise TypeError("Unsupported model output type for prediction extraction.")

    @staticmethod
    def _resolve_image_size(model_factory, checkpoint_payload: dict, fallback_size: int | None) -> int:
        return model_factory.resolve_image_size(checkpoint_payload, requested_image_size=fallback_size)

    def _load_model(self, model_name: str, checkpoint_path: Path):
        payload = self.model_factory.load_checkpoint_payload(checkpoint_path)
        image_size = self._resolve_image_size(self.model_factory, payload, self.config.image_size)
        try:
            model, _checkpoint, _resolved_size = self.model_factory.load_checkpoint(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                image_size=image_size,
                device=self.runtime.device,
            )
            model.eval()
            return model, image_size
        except AttributeError as error:
            # Some transformers versions can raise AttributeError instead of OSError
            # when local cache resolution fails for Dinov2Model.
            if model_name != "dino":
                raise
            raise RuntimeError(
                "Failed to load DINO backbone from local cache. "
                "Please ensure 'facebook/dinov2-small' is cached or run once with internet access."
            ) from error

    def _collect_eval_data(self, model_name: str, checkpoint_path: Path):
        model, image_size = self._load_model(model_name, checkpoint_path)
        dataset = VOCIoUTestDataset(self.config.test_voc_root, image_size=image_size)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            collate_fn=collate_iou_batch,
        )

        iou_by_class = defaultdict(list)
        all_predictions = []
        all_targets = []
        with torch.inference_mode():
            for images, targets in tqdm(loader, desc=f"{model_name} IoU", dynamic_ncols=True):
                images_device = [image.to(self.runtime.device) for image in images]
                targets_device = [
                    {k: (v.to(self.runtime.device) if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
                    for target in targets
                ]
                output = model(images_device, targets_device)
                predictions = self._extract_predictions(output)

                for prediction, target in zip(predictions, targets):
                    pred_boxes = prediction["boxes"].detach().cpu()
                    pred_scores = prediction["scores"].detach().cpu()
                    pred_labels = prediction["labels"].detach().cpu()
                    gt_boxes = target["boxes"].detach().cpu()
                    gt_labels = target["labels"].detach().cpu()
                    gt_difficult = target["difficult"].detach().cpu()

                    for class_idx, class_name in enumerate(VOC_CLASSES, start=1):
                        gt_class_boxes = gt_boxes[gt_labels == class_idx]
                        pred_class_boxes = pred_boxes[pred_labels == class_idx]
                        if gt_class_boxes.numel() == 0:
                            continue
                        if pred_class_boxes.numel() == 0:
                            iou_by_class[class_name].extend([0.0] * gt_class_boxes.shape[0])
                            continue
                        ious = box_iou(gt_class_boxes, pred_class_boxes)
                        best_per_gt = ious.max(dim=1).values
                        iou_by_class[class_name].extend(best_per_gt.tolist())

                    all_predictions.append(
                        {
                            "boxes": pred_boxes,
                            "scores": pred_scores,
                            "labels": pred_labels,
                        }
                    )
                    all_targets.append(
                        {
                            "boxes": gt_boxes,
                            "labels": gt_labels,
                            "difficult": gt_difficult,
                        }
                    )
        return iou_by_class, all_predictions, all_targets

    @staticmethod
    def _to_stats(values: list[float]) -> dict[str, float | int]:
        array = np.array(values, dtype=np.float32)
        if array.size == 0:
            return {"count": 0, "mean_iou": 0.0, "median_iou": 0.0, "p75_iou": 0.0}
        return {
            "count": int(array.size),
            "mean_iou": float(array.mean()),
            "median_iou": float(np.median(array)),
            "p75_iou": float(np.percentile(array, 75)),
        }

    def _save_stats(self, stats: dict) -> None:
        stats_path = self.output_dir / "iou_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    @staticmethod
    def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
        recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
        precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
        for index in range(precisions.numel() - 1, 0, -1):
            precisions[index - 1] = torch.maximum(precisions[index - 1], precisions[index])
        step_points = torch.where(recalls[1:] != recalls[:-1])[0]
        ap = torch.sum((recalls[step_points + 1] - recalls[step_points]) * precisions[step_points + 1])
        return float(ap.item())

    def _compute_pr_curve_for_class(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        class_idx: int,
        iou_threshold: float = 0.5,
    ) -> dict[str, float | list[float]]:
        class_predictions = []
        gt_by_image = defaultdict(list)
        gt_count = 0

        for image_idx, target in enumerate(targets):
            mask = target["labels"] == class_idx
            gt_boxes = target["boxes"][mask]
            difficult = target["difficult"][mask]
            for box, diff in zip(gt_boxes, difficult):
                diff_flag = bool(diff.item())
                gt_by_image[image_idx].append({"box": box, "matched": False, "difficult": diff_flag})
                if not diff_flag:
                    gt_count += 1

        for image_idx, prediction in enumerate(predictions):
            mask = prediction["labels"] == class_idx
            pred_boxes = prediction["boxes"][mask]
            pred_scores = prediction["scores"][mask]
            for box, score in zip(pred_boxes, pred_scores):
                class_predictions.append({"image_idx": image_idx, "box": box, "score": float(score.item())})

        class_predictions.sort(key=lambda item: item["score"], reverse=True)
        if gt_count == 0:
            return {"ap": 0.0, "recall": [0.0, 1.0], "precision": [0.0, 0.0], "gt_count": 0}

        true_positive = torch.zeros(len(class_predictions))
        false_positive = torch.zeros(len(class_predictions))

        for pred_idx, prediction in enumerate(class_predictions):
            candidates = gt_by_image[prediction["image_idx"]]
            if not candidates:
                false_positive[pred_idx] = 1
                continue

            candidate_boxes = torch.stack([entry["box"] for entry in candidates], dim=0)
            ious = box_iou(prediction["box"].unsqueeze(0), candidate_boxes).squeeze(0)
            best_iou, best_index = ious.max(dim=0)

            if best_iou >= iou_threshold:
                matched_entry = candidates[int(best_index.item())]
                if matched_entry["difficult"]:
                    continue
                if not matched_entry["matched"]:
                    true_positive[pred_idx] = 1
                    matched_entry["matched"] = True
                else:
                    false_positive[pred_idx] = 1
            else:
                false_positive[pred_idx] = 1

        tp_cum = torch.cumsum(true_positive, dim=0)
        fp_cum = torch.cumsum(false_positive, dim=0)
        recalls = tp_cum / max(gt_count, 1)
        precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)
        ap = self._compute_ap(recalls, precisions)
        return {
            "ap": ap,
            "recall": recalls.tolist(),
            "precision": precisions.tolist(),
            "gt_count": int(gt_count),
        }

    def _compute_pr_curves(self, eval_data: dict[str, dict]) -> dict[str, dict]:
        pr_results = {}
        for display_name in eval_data:
            predictions = eval_data[display_name]["predictions"]
            targets = eval_data[display_name]["targets"]
            pr_results[display_name] = {}
            for class_idx, class_name in enumerate(VOC_CLASSES, start=1):
                pr_results[display_name][class_name] = self._compute_pr_curve_for_class(
                    predictions=predictions,
                    targets=targets,
                    class_idx=class_idx,
                    iou_threshold=0.5,
                )
        return pr_results

    def _save_ap_stats(self, ap_stats: dict) -> None:
        stats_path = self.output_dir / "ap_stats.json"
        stats_path.write_text(json.dumps(ap_stats, indent=2), encoding="utf-8")

    def _plot_summary(self, stats: dict) -> None:
        classes = list(VOC_CLASSES)
        x = np.arange(len(classes))
        width = 0.35
        dino_means = [stats["DINO"][class_name]["mean_iou"] for class_name in classes]
        faster_means = [stats["Faster R-CNN"][class_name]["mean_iou"] for class_name in classes]

        fig, axis = plt.subplots(figsize=(9, 5))
        dino_bars = axis.bar(x - width / 2, dino_means, width, label="DINO", color="#1f77b4")
        faster_bars = axis.bar(x + width / 2, faster_means, width, label="Faster R-CNN", color="#ff7f0e")
        axis.set_xlabel("Class")
        axis.set_ylabel("Mean IoU")
        axis.set_title("Per-Class IoU Comparison (Test)")
        axis.set_xticks(x, classes)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

        for bars in (dino_bars, faster_bars):
            for bar in bars:
                value = bar.get_height()
                axis.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.tight_layout()
        fig.savefig(self.output_dir / "iou_per_class_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _plot_distributions(self, ious_by_model: dict[str, dict[str, list[float]]]) -> None:
        for class_name in VOC_CLASSES:
            dino_values = np.array(ious_by_model["DINO"].get(class_name, []), dtype=np.float32)
            faster_values = np.array(ious_by_model["Faster R-CNN"].get(class_name, []), dtype=np.float32)
            bins = np.linspace(0.0, 1.0, 21)

            fig, axis = plt.subplots(figsize=(9, 5))
            if dino_values.size:
                axis.hist(
                    dino_values,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="#1f77b4",
                    label=f"DINO (n={dino_values.size})",
                )
            if faster_values.size:
                axis.hist(
                    faster_values,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="#ff7f0e",
                    label=f"Faster R-CNN (n={faster_values.size})",
                )
            axis.set_xlim(0, 1.0)
            axis.set_xlabel("IoU")
            axis.set_ylabel("Density")
            axis.set_title(f"IoU Distribution - {class_name}")
            axis.grid(alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / f"iou_distribution_{class_name}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_pr_curves(self, pr_curves: dict[str, dict]) -> None:
        colors = {"DINO": "#1f77b4", "Faster R-CNN": "#ff7f0e"}
        for class_name in VOC_CLASSES:
            fig, axis = plt.subplots(figsize=(8, 6))
            for model_name in ("DINO", "Faster R-CNN"):
                curve = pr_curves[model_name][class_name]
                recall = np.array(curve["recall"], dtype=np.float32)
                precision = np.array(curve["precision"], dtype=np.float32)
                ap = float(curve["ap"])
                if recall.size == 0:
                    recall = np.array([0.0, 1.0], dtype=np.float32)
                    precision = np.array([0.0, 0.0], dtype=np.float32)
                axis.plot(
                    recall,
                    precision,
                    linewidth=2,
                    color=colors[model_name],
                    label=f"{model_name} (AP={ap:.3f})",
                )
                axis.fill_between(recall, precision, alpha=0.10, color=colors[model_name])

            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title(f"Precision-Recall Curve ({class_name})")
            axis.grid(alpha=0.3)
            axis.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"pr_curve_{class_name}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_ap_summary(self, pr_curves: dict[str, dict]) -> None:
        classes = list(VOC_CLASSES)
        x = np.arange(len(classes))
        width = 0.35
        dino_ap = [float(pr_curves["DINO"][name]["ap"]) for name in classes]
        faster_ap = [float(pr_curves["Faster R-CNN"][name]["ap"]) for name in classes]

        fig, axis = plt.subplots(figsize=(9, 5))
        bars_dino = axis.bar(x - width / 2, dino_ap, width, label="DINO", color="#1f77b4")
        bars_faster = axis.bar(x + width / 2, faster_ap, width, label="Faster R-CNN", color="#ff7f0e")
        axis.set_xticks(x, classes)
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("Class")
        axis.set_ylabel("AP (AUC of PR curve)")
        axis.set_title("Average Precision @ IoU=0.5")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

        for bars in (bars_dino, bars_faster):
            for bar in bars:
                value = bar.get_height()
                axis.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(self.output_dir / "ap_per_class_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def run(self) -> None:
        checkpoints = {
            "DINO": ("dino", Path(self.config.dino_checkpoint)),
            "Faster R-CNN": ("fasterrcnn", Path(self.config.fasterrcnn_checkpoint)),
        }
        for _, (_, path) in checkpoints.items():
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

        ious_by_model = {}
        eval_data = {}
        for display_name, (model_name, checkpoint_path) in checkpoints.items():
            iou_data, predictions, targets = self._collect_eval_data(model_name, checkpoint_path)
            ious_by_model[display_name] = iou_data
            eval_data[display_name] = {"predictions": predictions, "targets": targets}

        stats = {
            display_name: {
                class_name: self._to_stats(ious_by_model[display_name].get(class_name, []))
                for class_name in VOC_CLASSES
            }
            for display_name in checkpoints
        }
        pr_curves = self._compute_pr_curves(eval_data)
        self._save_stats(stats)
        self._save_ap_stats(pr_curves)
        self._plot_summary(stats)
        self._plot_distributions(ious_by_model)
        self._plot_pr_curves(pr_curves)
        self._plot_ap_summary(pr_curves)


def parse_args() -> IoUPlotConfig:
    parser = argparse.ArgumentParser(description="Generate read-only IoU plots from existing checkpoints and VOC2007 test data.")
    parser.add_argument("--test-voc-root", type=str, default="data/test-data/VOC2007")
    parser.add_argument("--dino-checkpoint", type=str, default="outputs/dino-final/best.pt")
    parser.add_argument("--fasterrcnn-checkpoint", type=str, default="outputs/fasterrcnn-final/best.pt")
    parser.add_argument("--image-size", type=int, default=None, help="Optional override. Uses checkpoint image_size when omitted.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="reports/plots")
    parser.add_argument("--hf-cache-dir", type=str, default=".hf_cache")
    args = parser.parse_args()
    return IoUPlotConfig(
        test_voc_root=args.test_voc_root,
        dino_checkpoint=args.dino_checkpoint,
        fasterrcnn_checkpoint=args.fasterrcnn_checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
    )


def main() -> None:
    IoUPlotGenerator(parse_args()).run()


if __name__ == "__main__":
    main()
