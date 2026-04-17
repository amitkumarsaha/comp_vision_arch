from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _series(history: list[dict], section: str, key: str) -> list[float]:
    values = []
    for row in history:
        values.append(float(row[section][key]))
    return values


def _epochs(history: list[dict]) -> list[int]:
    return [int(row["epoch"]) for row in history]


def _best_epoch(history: list[dict]) -> int:
    return max(history, key=lambda row: row["eval"]["mAP@0.5"])["epoch"]


@dataclass(frozen=True)
class PlotConfig:
    dino_summary: str
    faster_summary: str
    dino_progress: str
    faster_progress: str
    output_root: str


class ReportPlotGenerator:
    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        self.output_root = Path(config.output_root)
        self.training_dir = _ensure_dir(self.output_root / "training_stats")
        self.performance_dir = _ensure_dir(self.output_root / "performance_comparison")
        self.efficiency_dir = _ensure_dir(self.output_root / "efficiency_analysis")
        self.tables_dir = _ensure_dir(self.output_root / "report_tables")
        self.audit_dir = _ensure_dir(self.output_root / "audit")

        self.dino_summary = _load_json(config.dino_summary)
        self.faster_summary = _load_json(config.faster_summary)
        self.dino_progress = _load_json(config.dino_progress)
        self.faster_progress = _load_json(config.faster_progress)

        self.colors = {
            "dino": "#1f6feb",
            "faster": "#d97706",
            "person": "#0f766e",
            "car": "#b91c1c",
            "dog": "#6d28d9",
        }

    def run(self) -> None:
        generated = []
        generated.extend(self._plot_map_curve())
        generated.extend(self._plot_per_class_curves())
        generated.extend(self._plot_dino_loss_components())
        generated.extend(self._plot_faster_loss_components())
        generated.extend(self._plot_epoch_durations())
        generated.extend(self._plot_best_map_bar())
        generated.extend(self._plot_best_per_class_bar())
        generated.extend(self._plot_best_vs_final_bar())
        generated.extend(self._plot_efficiency_scatter())
        generated.extend(self._plot_ap_per_million_params())
        generated.extend(self._plot_summary_table())
        self._write_manifest(generated)

    def _save(self, figure: plt.Figure, path: Path) -> dict:
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return {"file": str(path.resolve()), "category": path.parent.name, "name": path.name}

    def _plot_map_curve(self) -> list[dict]:
        dino_history = self.dino_summary["history"]
        faster_history = self.faster_summary["history"]
        epochs = _epochs(dino_history)

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(epochs, _series(dino_history, "eval", "mAP@0.5"), marker="o", linewidth=2.5, color=self.colors["dino"], label="DINOv2 + custom head")
        axis.plot(epochs, _series(faster_history, "eval", "mAP@0.5"), marker="s", linewidth=2.5, color=self.colors["faster"], label="Faster R-CNN")
        axis.set_title("Validation mAP@0.5 Across Epochs")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("mAP@0.5")
        axis.grid(True, alpha=0.25)
        axis.legend()
        return [self._save(figure, self.training_dir / "map_curve_comparison.png")]

    def _plot_per_class_curves(self) -> list[dict]:
        dino_history = self.dino_summary["history"]
        faster_history = self.faster_summary["history"]
        epochs = _epochs(dino_history)
        classes = ["person", "car", "dog"]

        figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for axis, class_name in zip(axes, classes):
            axis.plot(epochs, _series(dino_history, "eval", class_name), marker="o", linewidth=2, color=self.colors["dino"], label="DINO")
            axis.plot(epochs, _series(faster_history, "eval", class_name), marker="s", linewidth=2, color=self.colors["faster"], label="Faster R-CNN")
            axis.set_title(f"{class_name.title()} AP@0.5")
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)
        axes[0].set_ylabel("AP@0.5")
        axes[0].legend()
        return [self._save(figure, self.training_dir / "per_class_ap_curves.png")]

    def _plot_dino_loss_components(self) -> list[dict]:
        history = self.dino_summary["history"]
        epochs = _epochs(history)
        keys = [
            ("loss", "Total loss"),
            ("loss_obj", "Objectness"),
            ("loss_cls", "Classification"),
            ("loss_l1", "Box L1"),
            ("loss_giou", "GIoU"),
        ]
        figure, axis = plt.subplots(figsize=(10, 6))
        for key, label in keys:
            axis.plot(epochs, _series(history, "train", key), marker="o", linewidth=2, label=label)
        axis.set_title("DINO Training Loss Components")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
        axis.legend()
        return [self._save(figure, self.training_dir / "dino_loss_components.png")]

    def _plot_faster_loss_components(self) -> list[dict]:
        history = self.faster_summary["history"]
        epochs = _epochs(history)
        keys = [
            ("loss_classifier", "Classifier"),
            ("loss_box_reg", "Box regression"),
            ("loss_objectness", "Objectness"),
            ("loss_rpn_box_reg", "RPN box regression"),
        ]
        figure, axis = plt.subplots(figsize=(10, 6))
        for key, label in keys:
            axis.plot(epochs, _series(history, "train", key), marker="o", linewidth=2, label=label)
        axis.set_title("Faster R-CNN Training Loss Components")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
        axis.legend()
        return [self._save(figure, self.training_dir / "fasterrcnn_loss_components.png")]

    def _plot_epoch_durations(self) -> list[dict]:
        dino_minutes = np.array(self.dino_progress["epoch_durations_seconds"], dtype=float) / 60.0
        faster_minutes = np.array(self.faster_progress["epoch_durations_seconds"], dtype=float) / 60.0
        epochs = np.arange(1, min(len(dino_minutes), len(faster_minutes)) + 1)

        figure, axis = plt.subplots(figsize=(10, 6))
        width = 0.38
        axis.bar(epochs - width / 2, dino_minutes[: len(epochs)], width=width, color=self.colors["dino"], label="DINO")
        axis.bar(epochs + width / 2, faster_minutes[: len(epochs)], width=width, color=self.colors["faster"], label="Faster R-CNN")
        axis.set_title("Epoch Duration Comparison")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Minutes")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        return [self._save(figure, self.training_dir / "epoch_duration_comparison.png")]

    def _plot_best_map_bar(self) -> list[dict]:
        labels = ["DINO", "Faster R-CNN"]
        values = [float(self.dino_summary["best_map_50"]), float(self.faster_summary["best_map_50"])]
        colors = [self.colors["dino"], self.colors["faster"]]

        figure, axis = plt.subplots(figsize=(8, 5))
        bars = axis.bar(labels, values, color=colors)
        axis.set_title("Best Validation mAP@0.5")
        axis.set_ylabel("mAP@0.5")
        axis.set_ylim(0, 1.0)
        axis.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
        return [self._save(figure, self.performance_dir / "best_map_bar.png")]

    def _plot_best_per_class_bar(self) -> list[dict]:
        dino_best = max(self.dino_summary["history"], key=lambda row: row["eval"]["mAP@0.5"])["eval"]
        faster_best = max(self.faster_summary["history"], key=lambda row: row["eval"]["mAP@0.5"])["eval"]
        classes = ["person", "car", "dog"]
        x = np.arange(len(classes))
        width = 0.36

        figure, axis = plt.subplots(figsize=(9, 5))
        dino_vals = [float(dino_best[name]) for name in classes]
        faster_vals = [float(faster_best[name]) for name in classes]
        axis.bar(x - width / 2, dino_vals, width=width, color=self.colors["dino"], label="DINO")
        axis.bar(x + width / 2, faster_vals, width=width, color=self.colors["faster"], label="Faster R-CNN")
        axis.set_xticks(x, [name.title() for name in classes])
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("AP@0.5")
        axis.set_title("Best Per-Class AP Comparison")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        return [self._save(figure, self.performance_dir / "best_per_class_ap_bar.png")]

    def _plot_best_vs_final_bar(self) -> list[dict]:
        labels = ["DINO best", "DINO final", "Faster best", "Faster final"]
        values = [
            float(self.dino_summary["best_map_50"]),
            float(self.dino_summary["history"][-1]["eval"]["mAP@0.5"]),
            float(self.faster_summary["best_map_50"]),
            float(self.faster_summary["history"][-1]["eval"]["mAP@0.5"]),
        ]
        colors = [self.colors["dino"], "#6aa6ff", self.colors["faster"], "#f5a344"]

        figure, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(labels, values, color=colors)
        axis.set_title("Best vs Final Validation mAP@0.5")
        axis.set_ylabel("mAP@0.5")
        axis.set_ylim(0, 1.0)
        axis.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
        return [self._save(figure, self.performance_dir / "best_vs_final_map_bar.png")]

    def _plot_efficiency_scatter(self) -> list[dict]:
        params = np.array(
            [
                float(self.dino_summary["trainable_parameters"]) / 1_000_000,
                float(self.faster_summary["trainable_parameters"]) / 1_000_000,
            ]
        )
        scores = np.array([float(self.dino_summary["best_map_50"]), float(self.faster_summary["best_map_50"])])
        labels = ["DINO", "Faster R-CNN"]
        colors = [self.colors["dino"], self.colors["faster"]]

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(params, scores, s=180, c=colors)
        for x, y, label in zip(params, scores, labels):
            axis.text(x + 0.6, y, label, va="center")
        axis.set_title("Accuracy vs Trainable Parameters")
        axis.set_xlabel("Trainable parameters (millions)")
        axis.set_ylabel("Best mAP@0.5")
        axis.grid(True, alpha=0.25)
        return [self._save(figure, self.efficiency_dir / "accuracy_vs_trainable_params.png")]

    def _plot_ap_per_million_params(self) -> list[dict]:
        labels = ["DINO", "Faster R-CNN"]
        values = [
            float(self.dino_summary["best_map_50"]) / (float(self.dino_summary["trainable_parameters"]) / 1_000_000),
            float(self.faster_summary["best_map_50"]) / (float(self.faster_summary["trainable_parameters"]) / 1_000_000),
        ]
        colors = [self.colors["dino"], self.colors["faster"]]

        figure, axis = plt.subplots(figsize=(8, 5))
        bars = axis.bar(labels, values, color=colors)
        axis.set_title("Best mAP@0.5 per Million Trainable Parameters")
        axis.set_ylabel("mAP@0.5 / million params")
        axis.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.03, f"{value:.3f}", ha="center", va="bottom")
        return [self._save(figure, self.efficiency_dir / "map_per_million_params.png")]

    def _plot_summary_table(self) -> list[dict]:
        dino_best = max(self.dino_summary["history"], key=lambda row: row["eval"]["mAP@0.5"])
        faster_best = max(self.faster_summary["history"], key=lambda row: row["eval"]["mAP@0.5"])
        headers = ["Model", "Best epoch", "Best mAP@0.5", "Person", "Car", "Dog", "Trainable params"]
        rows = [
            [
                "DINO",
                str(dino_best["epoch"]),
                f'{dino_best["eval"]["mAP@0.5"]:.3f}',
                f'{dino_best["eval"]["person"]:.3f}',
                f'{dino_best["eval"]["car"]:.3f}',
                f'{dino_best["eval"]["dog"]:.3f}',
                f'{int(self.dino_summary["trainable_parameters"]):,}',
            ],
            [
                "Faster R-CNN",
                str(faster_best["epoch"]),
                f'{faster_best["eval"]["mAP@0.5"]:.3f}',
                f'{faster_best["eval"]["person"]:.3f}',
                f'{faster_best["eval"]["car"]:.3f}',
                f'{faster_best["eval"]["dog"]:.3f}',
                f'{int(self.faster_summary["trainable_parameters"]):,}',
            ],
        ]

        figure, axis = plt.subplots(figsize=(11, 2.6))
        axis.axis("off")
        table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        axis.set_title("Experiment Summary Table", pad=14)
        return [self._save(figure, self.tables_dir / "experiment_summary_table.png")]

    def _write_manifest(self, generated: list[dict]) -> None:
        payload = {
            "dino_summary": str(Path(self.config.dino_summary).resolve()),
            "faster_summary": str(Path(self.config.faster_summary).resolve()),
            "dino_progress": str(Path(self.config.dino_progress).resolve()),
            "faster_progress": str(Path(self.config.faster_progress).resolve()),
            "output_root": str(self.output_root.resolve()),
            "generated_files": generated,
            "best_epochs": {
                "dino": _best_epoch(self.dino_summary["history"]),
                "fasterrcnn": _best_epoch(self.faster_summary["history"]),
            },
        }
        with (self.audit_dir / "report_plots_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(description="Generate report-ready comparison plots from saved training outputs.")
    parser.add_argument("--dino-summary", default="outputs/dino-final/summary.json")
    parser.add_argument("--faster-summary", default="outputs/fasterrcnn-final/summary.json")
    parser.add_argument("--dino-progress", default="outputs/dino-final/audit/training_progress.json")
    parser.add_argument("--faster-progress", default="outputs/fasterrcnn-final/audit/training_progress.json")
    parser.add_argument("--output-root", default="outputs/viz")
    args = parser.parse_args()
    return PlotConfig(
        dino_summary=args.dino_summary,
        faster_summary=args.faster_summary,
        dino_progress=args.dino_progress,
        faster_progress=args.faster_progress,
        output_root=args.output_root,
    )


def main() -> None:
    ReportPlotGenerator(parse_args()).run()


if __name__ == "__main__":
    main()
