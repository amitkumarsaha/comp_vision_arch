from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import VOC_CLASSES


@dataclass(frozen=True)
class PlotConfig:
    dino_summary: str
    faster_summary: str
    dino_eval_data: str
    faster_eval_data: str
    output_dir: str


class ReportPlotGenerator:
    DINO_COLOR = "#1f77b4"
    FASTER_COLOR = "#ff7f0e"

    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dino_summary = self._load_json(config.dino_summary)
        self.faster_summary = self._load_json(config.faster_summary)
        self.dino_eval_data = self._load_json(config.dino_eval_data)
        self.faster_eval_data = self._load_json(config.faster_eval_data)

    @staticmethod
    def _load_json(path: str | Path) -> dict:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _epochs(history: list[dict]) -> list[int]:
        return [int(row["epoch"]) for row in history]

    @staticmethod
    def _series(history: list[dict], section: str, key: str) -> list[float]:
        return [float(row[section][key]) for row in history]

    @staticmethod
    def _pr_curve(eval_data: dict, class_name: str) -> tuple[np.ndarray, np.ndarray, float]:
        curve = (
            eval_data.get("plot_data", {})
            .get("pr_curves", {})
            .get(class_name, {})
        )
        recall = np.array(curve.get("recall", [0.0, 1.0]), dtype=np.float32)
        precision = np.array(curve.get("precision", [0.0, 0.0]), dtype=np.float32)
        ap = float(curve.get("ap", 0.0))
        return recall, precision, ap

    @staticmethod
    def _iou_values(eval_data: dict, class_name: str) -> np.ndarray:
        values = (
            eval_data.get("plot_data", {})
            .get("iou_distribution", {})
            .get(class_name, [])
        )
        return np.array(values, dtype=np.float32)

    def _plot_map_single_axis(self) -> None:
        d_hist = self.dino_summary["history"]
        f_hist = self.faster_summary["history"]
        epochs = self._epochs(d_hist)
        d_map = self._series(d_hist, "eval", "mAP@0.5")
        f_map = self._series(f_hist, "eval", "mAP@0.5")

        fig, axis = plt.subplots(figsize=(10, 5))
        axis.plot(epochs, d_map, color=self.DINO_COLOR, marker="o", linewidth=2, label="DINO")
        axis.plot(epochs, f_map, color=self.FASTER_COLOR, marker="s", linewidth=2, label="Faster R-CNN")

        d_best = max(d_map)
        f_best = max(f_map)
        axis.axhline(d_best, color=self.DINO_COLOR, linestyle="--", linewidth=1.5, alpha=0.85)
        axis.axhline(f_best, color=self.FASTER_COLOR, linestyle="--", linewidth=1.5, alpha=0.85)
        axis.annotate(
            f"Best DINO: {d_best:.3f}",
            xy=(epochs[d_map.index(d_best)], d_best),
            xytext=(8, 10),
            textcoords="offset points",
            color=self.DINO_COLOR,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=self.DINO_COLOR,
                linewidth=1.0,
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="->", color=self.DINO_COLOR, lw=1.1),
        )
        axis.annotate(
            f"Best Faster R-CNN: {f_best:.3f}",
            xy=(epochs[f_map.index(f_best)], f_best),
            xytext=(8, -32),
            textcoords="offset points",
            color=self.FASTER_COLOR,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=self.FASTER_COLOR,
                linewidth=1.0,
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="->", color=self.FASTER_COLOR, lw=1.1),
        )

        axis.set_title("mAP@0.5 over Epochs (Single Axis)")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("mAP@0.5")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "map_single_axis_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _plot_ap_per_class_annotated(self) -> None:
        classes = ["person", "car", "dog"]
        d_hist = self.dino_summary["history"]
        f_hist = self.faster_summary["history"]
        epochs = self._epochs(d_hist)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for axis, cls in zip(axes, classes):
            d_vals = self._series(d_hist, "eval", cls)
            f_vals = self._series(f_hist, "eval", cls)
            axis.plot(epochs, d_vals, color=self.DINO_COLOR, marker="o", linewidth=2, label="DINO")
            axis.plot(epochs, f_vals, color=self.FASTER_COLOR, marker="s", linewidth=2, label="Faster R-CNN")

            d_best = max(d_vals)
            f_best = max(f_vals)
            d_idx = d_vals.index(d_best)
            f_idx = f_vals.index(f_best)
            d_x = epochs[d_idx]
            f_x = epochs[f_idx]

            axis.axhline(d_best, color=self.DINO_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
            axis.axhline(f_best, color=self.FASTER_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)

            if cls == "car":
                d_text_x = max(min(epochs) + 0.4, d_x - 2.4)
                d_text_ha = "right"
            elif cls == "dog":
                d_text_x = max(min(epochs) + 0.8, d_x - 1.3)
                d_text_ha = "right"
            else:
                d_text_x = d_x + 0.45 if d_x < max(epochs) - 1 else d_x - 2.2
                d_text_ha = "left" if d_text_x >= d_x else "right"
            d_text_y = min(0.95, d_best + 0.06) if d_best <= 0.9 else max(0.05, d_best - 0.08)

            axis.annotate(
                f"Best DINO AP@0.5: {d_best:.3f}",
                xy=(d_x, d_best),
                xycoords="data",
                xytext=(d_text_x, d_text_y),
                textcoords="data",
                ha=d_text_ha,
                va="bottom" if d_text_y >= d_best else "top",
                fontsize=8,
                color=self.DINO_COLOR,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=self.DINO_COLOR, linewidth=1.0, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=self.DINO_COLOR, lw=1.0, alpha=0.9),
                annotation_clip=True,
            )

            f_text_x = f_x + 0.45 if f_x < max(epochs) - 1 else f_x - 2.2
            if cls == "dog":
                f_text_y = min(0.95, f_best + 0.07)
                f_va = "bottom"
            else:
                f_text_y = max(0.05, f_best - 0.08 if f_best > 0.88 else f_best - 0.06)
                f_va = "top"
            axis.annotate(
                f"Best Faster R-CNN AP@0.5: {f_best:.3f}",
                xy=(f_x, f_best),
                xycoords="data",
                xytext=(f_text_x, f_text_y),
                textcoords="data",
                ha="left" if f_text_x >= f_x else "right",
                va=f_va,
                fontsize=8,
                color=self.FASTER_COLOR,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=self.FASTER_COLOR, linewidth=1.0, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=self.FASTER_COLOR, lw=1.0, alpha=0.9),
            )

            axis.set_title(f"AP@0.5 - {cls}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("AP@0.5")
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axis.grid(True, alpha=0.3)
            axis.legend(loc="lower right", frameon=True)

        fig.tight_layout()
        fig.savefig(self.output_dir / "ap_per_class_over_epochs_annotated.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _plot_training_losses_combined(self) -> None:
        d_hist = self.dino_summary["history"]
        f_hist = self.faster_summary["history"]
        epochs = self._epochs(d_hist)

        dino_keys = ["loss", "loss_obj", "loss_cls", "loss_l1", "loss_giou"]
        dino_series = {}
        for key in dino_keys:
            if all(key in row.get("train", {}) for row in d_hist):
                dino_series[key] = self._series(d_hist, "train", key)

        faster_keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
        faster_series = {}
        for key in faster_keys:
            if all(key in row.get("train", {}) for row in f_hist):
                faster_series[key] = self._series(f_hist, "train", key)

        d_total = dino_series.get("loss", [])
        if not d_total:
            d_total = [
                float(row["train"].get("loss_obj", 0.0))
                + float(row["train"].get("loss_cls", 0.0))
                + float(row["train"].get("loss_l1", 0.0))
                + float(row["train"].get("loss_giou", 0.0))
                for row in d_hist
            ]
        f_total = [
            float(row["train"].get("loss_classifier", 0.0))
            + float(row["train"].get("loss_box_reg", 0.0))
            + float(row["train"].get("loss_objectness", 0.0))
            + float(row["train"].get("loss_rpn_box_reg", 0.0))
            for row in f_hist
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

        for key, values in dino_series.items():
            axes[0].plot(epochs, values, marker="o", linewidth=1.4, label=key)
        axes[0].set_title("DINO Training Loss Components")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for key, values in faster_series.items():
            axes[1].plot(epochs, values, marker="o", linewidth=1.4, label=key)
        axes[1].set_title("Faster R-CNN Training Loss Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, d_total, color=self.DINO_COLOR, marker="o", linewidth=1.4, label="DINO total train loss (sum)")
        axes[2].plot(epochs, f_total, color=self.FASTER_COLOR, marker="o", linewidth=1.4, label="Faster R-CNN total train loss (sum)")
        axes[2].set_title("Total Training Loss Proxy Over Epochs")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Total Training Loss (sum of components)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "training_losses_combined.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _plot_pr_curve_combined(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for axis, cls in zip(axes, VOC_CLASSES):
            d_recall, d_precision, d_ap = self._pr_curve(self.dino_eval_data, cls)
            f_recall, f_precision, f_ap = self._pr_curve(self.faster_eval_data, cls)

            axis.plot(d_recall, d_precision, linewidth=2, color=self.DINO_COLOR, label=f"DINO (AP={d_ap:.3f})")
            axis.fill_between(d_recall, d_precision, alpha=0.10, color=self.DINO_COLOR)
            axis.plot(f_recall, f_precision, linewidth=2, color=self.FASTER_COLOR, label=f"Faster R-CNN (AP={f_ap:.3f})")
            axis.fill_between(f_recall, f_precision, alpha=0.10, color=self.FASTER_COLOR)

            axis.set_title(f"PR Curve - {cls}")
            axis.set_xlabel("Recall")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="lower left")
        axes[0].set_ylabel("Precision")
        fig.tight_layout()
        fig.savefig(self.output_dir / "pr_curve_combined.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _plot_iou_distribution_combined(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        bins = np.linspace(0.0, 1.0, 21)
        for axis, cls in zip(axes, VOC_CLASSES):
            d_iou = self._iou_values(self.dino_eval_data, cls)
            f_iou = self._iou_values(self.faster_eval_data, cls)

            if d_iou.size:
                axis.hist(d_iou, bins=bins, density=True, alpha=0.5, color=self.DINO_COLOR, label=f"DINO (n={d_iou.size})")
            if f_iou.size:
                axis.hist(f_iou, bins=bins, density=True, alpha=0.5, color=self.FASTER_COLOR, label=f"Faster R-CNN (n={f_iou.size})")

            axis.set_title(f"IoU Distribution - {cls}")
            axis.set_xlabel("IoU")
            axis.set_xlim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.legend()
        axes[0].set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(self.output_dir / "iou_distribution_combined.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def run(self) -> None:
        self._plot_map_single_axis()
        self._plot_ap_per_class_annotated()
        self._plot_training_losses_combined()
        self._plot_pr_curve_combined()
        self._plot_iou_distribution_combined()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report plots from precomputed training and evaluation artifacts."
    )
    parser.add_argument("--dino-summary", type=str, default="outputs/dino-final/summary.json")
    parser.add_argument("--faster-summary", type=str, default="outputs/fasterrcnn-final/summary.json")
    parser.add_argument("--dino-eval-data", type=str, default="outputs/dino-final/audit/evaluation_plot_data.json")
    parser.add_argument("--faster-eval-data", type=str, default="outputs/fasterrcnn-final/audit/evaluation_plot_data.json")
    parser.add_argument("--output-dir", type=str, default="output/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlotConfig(
        dino_summary=args.dino_summary,
        faster_summary=args.faster_summary,
        dino_eval_data=args.dino_eval_data,
        faster_eval_data=args.faster_eval_data,
        output_dir=args.output_dir,
    )
    ReportPlotGenerator(config).run()


if __name__ == "__main__":
    main()
