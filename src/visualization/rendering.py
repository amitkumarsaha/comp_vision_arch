from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from ..utils import IDX_TO_CLASS

CLASS_COLORS = {
    "person": "#ff4d4f",
    "car": "#1890ff",
    "dog": "#52c41a",
}
GROUND_TRUTH_COLOR = "#fadb14"


class PredictionAdapter:
    @staticmethod
    def extract(output):
        predictions = output if isinstance(output, list) else output.predictions
        return [{key: value.detach().cpu() for key, value in prediction.items()} for prediction in predictions]


class PredictionRenderer:
    def draw_prediction_on_axis(self, axis, image: torch.Tensor, prediction: dict[str, torch.Tensor], title: str):
        axis.imshow(image.permute(1, 2, 0).cpu().numpy())
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            x1, y1, x2, y2 = box.tolist()
            class_name = IDX_TO_CLASS[int(label)]
            color = CLASS_COLORS.get(class_name, "#ffd666")
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=3, edgecolor=color))
            axis.text(
                x1,
                max(y1 - 4, 2),
                f"{class_name}: {float(score):.2f}",
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": color, "edgecolor": color, "boxstyle": "round,pad=0.2"},
            )
        axis.set_title(title)
        axis.axis("off")

    def draw_ground_truth_on_axis(self, axis, image: torch.Tensor, target: dict[str, torch.Tensor], title: str):
        axis.imshow(image.permute(1, 2, 0).cpu().numpy())
        for box, label in zip(target["boxes"].detach().cpu(), target["labels"].detach().cpu()):
            x1, y1, x2, y2 = box.tolist()
            class_name = IDX_TO_CLASS[int(label)]
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=3,
                    edgecolor=GROUND_TRUTH_COLOR,
                    linestyle="--",
                )
            )
            axis.text(
                x1,
                max(y1 - 4, 2),
                f"GT {class_name}",
                color="black",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": GROUND_TRUTH_COLOR, "edgecolor": GROUND_TRUTH_COLOR, "boxstyle": "round,pad=0.2"},
            )
        axis.set_title(title)
        axis.axis("off")

    def save_single_prediction(self, image: torch.Tensor, prediction: dict[str, torch.Tensor], output_path: Path, title: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 8))
        self.draw_prediction_on_axis(axis, image, prediction, title)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def save_pair(self, image: torch.Tensor, dino_prediction: dict[str, torch.Tensor], fasterrcnn_prediction: dict[str, torch.Tensor], output_path: Path) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(16, 8))
        self.draw_prediction_on_axis(axes[0], image, dino_prediction, "DINO + Custom Head")
        self.draw_prediction_on_axis(axes[1], image, fasterrcnn_prediction, "Faster R-CNN")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def save_triptych(
        self,
        image: torch.Tensor,
        target: dict[str, torch.Tensor],
        dino_prediction: dict[str, torch.Tensor],
        fasterrcnn_prediction: dict[str, torch.Tensor],
        output_path: Path,
    ) -> None:
        figure, axes = plt.subplots(1, 3, figsize=(24, 8))
        self.draw_ground_truth_on_axis(axes[0], image, target, "Ground Truth")
        self.draw_prediction_on_axis(axes[1], image, dino_prediction, "DINO + Custom Head")
        self.draw_prediction_on_axis(axes[2], image, fasterrcnn_prediction, "Faster R-CNN")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def save_comparison_grid(self, rows: list[dict], output_path: Path) -> None:
        if not rows:
            return
        figure, axes = plt.subplots(len(rows), 3, figsize=(24, 6 * len(rows)))
        if len(rows) == 1:
            axes = [axes]
        for row_axes, row in zip(axes, rows):
            self.draw_ground_truth_on_axis(row_axes[0], row["image"], row["target"], f"Ground Truth | {row['image_id']}")
            self.draw_prediction_on_axis(row_axes[1], row["image"], row["dino_prediction"], f"DINO + Custom Head | {row['image_id']}")
            self.draw_prediction_on_axis(row_axes[2], row["image"], row["fasterrcnn_prediction"], f"Faster R-CNN | {row['image_id']}")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
