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
class APPlotConfig:
    test_voc_root: str
    dino_checkpoint: str
    fasterrcnn_checkpoint: str
    image_size: int | None
    batch_size: int
    workers: int
    output_dir: str
    hf_cache_dir: str


class VOCTestThreeClass(Dataset):
    def __init__(self, voc2007_root: str | Path, image_size: int) -> None:
        self.voc_root = Path(voc2007_root)
        self.image_size = int(image_size)
        split_file = self.voc_root / "ImageSets" / "Main" / "test.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing VOC test split file: {split_file}")
        image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_ids = [image_id for image_id in image_ids if self._has_target_class(image_id)]

    def _annotation_path(self, image_id: str) -> Path:
        return self.voc_root / "Annotations" / f"{image_id}.xml"

    def _image_path(self, image_id: str) -> Path:
        return self.voc_root / "JPEGImages" / f"{image_id}.jpg"

    def _parse_objects(self, image_id: str):
        root = ET.parse(self._annotation_path(image_id)).getroot()
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
        }
        return image_tensor, target


def collate_batch(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class APPlotGenerator:
    def __init__(self, config: APPlotConfig) -> None:
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

    def _load_model(self, model_name: str, checkpoint_path: Path):
        payload = self.model_factory.load_checkpoint_payload(checkpoint_path)
        image_size = self.model_factory.resolve_image_size(payload, requested_image_size=self.config.image_size)
        model, _checkpoint, _resolved_size = self.model_factory.load_checkpoint(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            device=self.runtime.device,
        )
        model.eval()
        return model, image_size

    def _collect_predictions(self, model_name: str, checkpoint_path: Path):
        model, image_size = self._load_model(model_name, checkpoint_path)
        dataset = VOCTestThreeClass(self.config.test_voc_root, image_size=image_size)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            collate_fn=collate_batch,
        )

        predictions = []
        targets = []
        with torch.inference_mode():
            for images, batch_targets in tqdm(loader, desc=f"{model_name} AP", dynamic_ncols=True):
                images_device = [img.to(self.runtime.device) for img in images]
                targets_device = [
                    {k: (v.to(self.runtime.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                    for t in batch_targets
                ]
                output = model(images_device, targets_device)
                batch_predictions = self._extract_predictions(output)

                for prediction, target in zip(batch_predictions, batch_targets):
                    predictions.append(
                        {
                            "boxes": prediction["boxes"].detach().cpu(),
                            "scores": prediction["scores"].detach().cpu(),
                            "labels": prediction["labels"].detach().cpu(),
                        }
                    )
                    targets.append(
                        {
                            "boxes": target["boxes"].detach().cpu(),
                            "labels": target["labels"].detach().cpu(),
                            "difficult": target["difficult"].detach().cpu(),
                        }
                    )
        return predictions, targets

    @staticmethod
    def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
        recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
        precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
        for idx in range(precisions.numel() - 1, 0, -1):
            precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])
        step_points = torch.where(recalls[1:] != recalls[:-1])[0]
        ap = torch.sum((recalls[step_points + 1] - recalls[step_points]) * precisions[step_points + 1])
        return float(ap.item())

    def _curve_for_class(
        self,
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
                is_difficult = bool(diff.item())
                gt_by_image[image_idx].append({"box": box, "matched": False, "difficult": is_difficult})
                if not is_difficult:
                    gt_count += 1

        for image_idx, prediction in enumerate(predictions):
            pred_mask = prediction["labels"] == class_idx
            pred_boxes = prediction["boxes"][pred_mask]
            pred_scores = prediction["scores"][pred_mask]
            for box, score in zip(pred_boxes, pred_scores):
                class_predictions.append({"image_idx": image_idx, "box": box, "score": float(score.item())})

        class_predictions.sort(key=lambda item: item["score"], reverse=True)
        if gt_count == 0:
            return {"ap": 0.0, "recall": [0.0, 1.0], "precision": [0.0, 0.0], "gt_count": 0}

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
        recalls = tp_cum / max(gt_count, 1)
        precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)
        ap = self._compute_ap(recalls, precisions)
        return {
            "ap": ap,
            "recall": recalls.tolist(),
            "precision": precisions.tolist(),
            "gt_count": int(gt_count),
        }

    def _compute_pr_curves(self, data_by_model: dict[str, dict]) -> dict[str, dict]:
        curves = {}
        for model_name, eval_data in data_by_model.items():
            curves[model_name] = {}
            for class_idx, class_name in enumerate(VOC_CLASSES, start=1):
                curves[model_name][class_name] = self._curve_for_class(
                    predictions=eval_data["predictions"],
                    targets=eval_data["targets"],
                    class_idx=class_idx,
                    iou_threshold=0.5,
                )
        return curves

    def _save_stats(self, curves: dict[str, dict]) -> None:
        (self.output_dir / "ap_stats.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")

    def _plot_pr_curves(self, curves: dict[str, dict]) -> None:
        colors = {"DINO": "#1f77b4", "Faster R-CNN": "#ff7f0e"}
        for class_name in VOC_CLASSES:
            fig, axis = plt.subplots(figsize=(8, 6))
            for model_name in ("DINO", "Faster R-CNN"):
                curve = curves[model_name][class_name]
                recall = np.array(curve["recall"], dtype=np.float32)
                precision = np.array(curve["precision"], dtype=np.float32)
                ap = float(curve["ap"])
                if recall.size == 0:
                    recall = np.array([0.0, 1.0], dtype=np.float32)
                    precision = np.array([0.0, 0.0], dtype=np.float32)
                axis.plot(recall, precision, linewidth=2, color=colors[model_name], label=f"{model_name} (AP={ap:.3f})")
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

    def _plot_ap_summary(self, curves: dict[str, dict]) -> None:
        classes = list(VOC_CLASSES)
        x = np.arange(len(classes))
        width = 0.35
        dino_ap = [float(curves["DINO"][name]["ap"]) for name in classes]
        faster_ap = [float(curves["Faster R-CNN"][name]["ap"]) for name in classes]

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

        eval_data = {}
        for display_name, (model_name, checkpoint_path) in checkpoints.items():
            predictions, targets = self._collect_predictions(model_name, checkpoint_path)
            eval_data[display_name] = {"predictions": predictions, "targets": targets}

        curves = self._compute_pr_curves(eval_data)
        self._save_stats(curves)
        self._plot_pr_curves(curves)
        self._plot_ap_summary(curves)


def parse_args() -> APPlotConfig:
    parser = argparse.ArgumentParser(description="Generate AP/PR plots from existing checkpoints and VOC2007 test data.")
    parser.add_argument("--test-voc-root", type=str, default="data/test-data/VOC2007")
    parser.add_argument("--dino-checkpoint", type=str, default="outputs/dino-final/best.pt")
    parser.add_argument("--fasterrcnn-checkpoint", type=str, default="outputs/fasterrcnn-final/best.pt")
    parser.add_argument("--image-size", type=int, default=None, help="Optional override. Uses checkpoint image_size when omitted.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="reports/plots")
    parser.add_argument("--hf-cache-dir", type=str, default=".hf_cache")
    args = parser.parse_args()
    return APPlotConfig(
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
    APPlotGenerator(parse_args()).run()


if __name__ == "__main__":
    main()
