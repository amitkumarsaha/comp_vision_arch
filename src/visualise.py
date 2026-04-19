from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
from src.utils import utc_timestamp, project_path
from src.visualization.rendering import PredictionAdapter, PredictionRenderer


@dataclass(frozen=True)
class VisualiseConfig:
    model: str | None
    train_data_root: str
    test_data_root: str
    checkpoint: str | None
    dino_checkpoint: str | None
    fasterrcnn_checkpoint: str | None
    output_dir: str
    num_images: int
    image_size: int


def parse_args():
    parser = argparse.ArgumentParser(description="Render predictions for a few test images.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], default=None)
    parser.add_argument("--train-data-root", type=str, default="data/train-validation-data")
    parser.add_argument("--test-data-root", type=str, default="data/test-data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--fasterrcnn-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    single_model_mode = args.model is not None and args.checkpoint is not None
    comparison_mode = args.dino_checkpoint is not None and args.fasterrcnn_checkpoint is not None
    if not single_model_mode and not comparison_mode:
        parser.error(
            "Use either --model with --checkpoint for single-model visualization, "
            "or use both --dino-checkpoint and --fasterrcnn-checkpoint for side-by-side comparison."
        )
    return args




class VisualisationApp:
    DINO_COLOR = "#1f77b4"
    FASTER_COLOR = "#ff7f0e"

    def __init__(self, config: VisualiseConfig) -> None:
        self.runtime = RuntimeEnvironment()
        resolved_image_size = config.image_size
        if config.checkpoint is not None:
            checkpoint = ModelFactory.load_checkpoint_payload(config.checkpoint)
            resolved_image_size = ModelFactory.resolve_image_size(checkpoint, config.image_size)
        elif config.dino_checkpoint is not None and config.fasterrcnn_checkpoint is not None:
            dino_checkpoint = ModelFactory.load_checkpoint_payload(config.dino_checkpoint)
            fasterrcnn_checkpoint = ModelFactory.load_checkpoint_payload(config.fasterrcnn_checkpoint)
            dino_image_size = ModelFactory.resolve_image_size(dino_checkpoint, config.image_size)
            fasterrcnn_image_size = ModelFactory.resolve_image_size(fasterrcnn_checkpoint, config.image_size)
            if dino_image_size != fasterrcnn_image_size:
                raise ValueError(
                    "Comparison checkpoints were trained with different image sizes "
                    f"({dino_image_size} vs {fasterrcnn_image_size})."
                )
            resolved_image_size = dino_image_size
        elif config.dino_checkpoint is not None:
            checkpoint = ModelFactory.load_checkpoint_payload(config.dino_checkpoint)
            resolved_image_size = ModelFactory.resolve_image_size(checkpoint, config.image_size)
        self.config = VisualiseConfig(
            model=config.model,
            train_data_root=config.train_data_root,
            test_data_root=config.test_data_root,
            checkpoint=config.checkpoint,
            dino_checkpoint=config.dino_checkpoint,
            fasterrcnn_checkpoint=config.fasterrcnn_checkpoint,
            output_dir=config.output_dir,
            num_images=config.num_images,
            image_size=resolved_image_size,
        )
        self.audit = AuditLogger(config.output_dir)
        self.data = DataPipelineManager(
            DataConfig(
                train_data_root=self.config.train_data_root,
                test_data_root=self.config.test_data_root,
                image_size=self.config.image_size,
                batch_size=1,
                workers=0,
                subset_size=None,
                seed=42,
            )
        )
        self.renderer = PredictionRenderer()

    def run(self) -> None:
        test_dataset = self.data.test_dataset()
        test_loader = self.data.test_loader(batch_size=1, workers=0)
        written_files = []
        comparison_rows = []

        if self._is_comparison_mode:
            self.config = VisualiseConfig(**{**self.config.__dict__, "model": "comparison"})
            dino_model, _, _ = ModelFactory.load_checkpoint("dino", self.config.dino_checkpoint, self.config.image_size, self.runtime.device)
            fasterrcnn_model, _, _ = ModelFactory.load_checkpoint("fasterrcnn", self.config.fasterrcnn_checkpoint, self.config.image_size, self.runtime.device)
            dino_model.eval()
            fasterrcnn_model.eval()
            audit_model = dino_model
        else:
            model, _, _ = ModelFactory.load_checkpoint(self.config.model, self.config.checkpoint, self.config.image_size, self.runtime.device)
            model.eval()
            audit_model = model

        saved = 0
        for images, targets, meta in test_loader:
            images = [image.to(self.runtime.device) for image in images]
            image_cpu = images[0].cpu()
            target_cpu = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in targets[0].items()}

            if self.config.model == "comparison":
                dino_prediction = PredictionAdapter.extract(dino_model(images, targets))[0]
                fasterrcnn_prediction = PredictionAdapter.extract(fasterrcnn_model(images, targets))[0]
                self._write_comparison_outputs(image_cpu, target_cpu, dino_prediction, fasterrcnn_prediction, meta[0].image_id, written_files, comparison_rows)
            else:
                prediction = PredictionAdapter.extract(model(images, targets))[0]
                image_path = self.audit.output_dir / f"{meta[0].image_id}.png"
                self.renderer.save_single_prediction(image_cpu, prediction, image_path, title=self.config.model.upper())
                written_files.append(
                    {
                        "image_id": meta[0].image_id,
                        "file": project_path(image_path),
                        "mode": self.config.model,
                        "prediction_count": int(prediction["boxes"].shape[0]),
                    }
                )
            saved += 1
            if saved >= self.config.num_images:
                break

        if self.config.model == "comparison" and comparison_rows:
            grid_path = self.audit.output_dir / "comparison_grid.png"
            self.renderer.save_comparison_grid(comparison_rows, grid_path)
            written_files.append({"image_id": "all", "file": project_path(grid_path), "mode": "comparison_grid", "rows": len(comparison_rows)})
            map_plot = self._write_map_single_axis_plot()
            if map_plot is not None:
                written_files.append(
                    {
                        "image_id": "all",
                        "file": project_path(map_plot),
                        "mode": "map_single_axis_comparison",
                    }
                )

        self.audit.write_runtime("visualization_run_manifest.json", self.config, audit_model, self.runtime.device)
        self.audit.write_dataset(
            "visualization_dataset_manifest.json",
            test_dataset=test_dataset,
            test_root=self.config.test_data_root,
            subset_size=None,
            seed=42,
        )
        self.audit.write_record(
            "visualization_manifest.json",
            {
                "timestamp_utc": utc_timestamp(),
                "mode": self.config.model,
                "checkpoint_path": project_path(self.config.checkpoint) if self.config.checkpoint else None,
                "dino_checkpoint_path": project_path(self.config.dino_checkpoint) if self.config.dino_checkpoint else None,
                "fasterrcnn_checkpoint_path": project_path(self.config.fasterrcnn_checkpoint) if self.config.fasterrcnn_checkpoint else None,
                "output_dir": project_path(self.audit.output_dir),
                "files": written_files,
            },
        )

    @property
    def _is_comparison_mode(self) -> bool:
        return self.config.dino_checkpoint is not None and self.config.fasterrcnn_checkpoint is not None

    def _write_comparison_outputs(
        self,
        image: torch.Tensor,
        target: dict[str, torch.Tensor],
        dino_prediction: dict[str, torch.Tensor],
        fasterrcnn_prediction: dict[str, torch.Tensor],
        image_id: str,
        written_files: list[dict],
        comparison_rows: list[dict],
    ) -> None:
        image_path = self.audit.output_dir / f"comparison_{image_id}.png"
        diagnostic_path = self.audit.output_dir / f"diagnostic_{image_id}.png"
        self.renderer.save_pair(image, dino_prediction, fasterrcnn_prediction, image_path)
        self.renderer.save_triptych(image, target, dino_prediction, fasterrcnn_prediction, diagnostic_path)
        comparison_rows.append(
            {
                "image_id": image_id,
                "image": image,
                "target": target,
                "dino_prediction": dino_prediction,
                "fasterrcnn_prediction": fasterrcnn_prediction,
            }
        )
        written_files.append(
            {
                "image_id": image_id,
                "file": project_path(image_path),
                "mode": "comparison",
                "dino_prediction_count": int(dino_prediction["boxes"].shape[0]),
                "fasterrcnn_prediction_count": int(fasterrcnn_prediction["boxes"].shape[0]),
            }
        )
        written_files.append(
            {
                "image_id": image_id,
                "file": project_path(diagnostic_path),
                "mode": "diagnostic_triptych",
                "ground_truth_count": int(target["boxes"].shape[0]),
                "dino_prediction_count": int(dino_prediction["boxes"].shape[0]),
                "fasterrcnn_prediction_count": int(fasterrcnn_prediction["boxes"].shape[0]),
            }
        )

    def _summary_path_from_checkpoint(self, checkpoint_path: str | None) -> Path | None:
        if checkpoint_path is None:
            return None
        return Path(checkpoint_path).resolve().parent / "summary.json"

    def _load_history(self, summary_path: Path | None) -> list[dict] | None:
        if summary_path is None or not summary_path.exists():
            return None
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        history = payload.get("history")
        if not isinstance(history, list) or not history:
            return None
        return history

    def _write_map_single_axis_plot(self) -> Path | None:
        dino_history = self._load_history(self._summary_path_from_checkpoint(self.config.dino_checkpoint))
        faster_history = self._load_history(self._summary_path_from_checkpoint(self.config.fasterrcnn_checkpoint))
        if dino_history is None or faster_history is None:
            return None

        epochs = [int(row["epoch"]) for row in dino_history]
        dino_map = [float(row["eval"]["mAP@0.5"]) for row in dino_history]
        faster_map = [float(row["eval"]["mAP@0.5"]) for row in faster_history]

        d_best = max(dino_map)
        f_best = max(faster_map)
        d_best_x = epochs[dino_map.index(d_best)]
        f_best_x = epochs[faster_map.index(f_best)]

        fig, axis = plt.subplots(figsize=(10, 5))
        axis.plot(epochs, dino_map, color=self.DINO_COLOR, marker="o", linewidth=2, label="DINO")
        axis.plot(epochs, faster_map, color=self.FASTER_COLOR, marker="s", linewidth=2, label="Faster R-CNN")
        axis.axhline(d_best, color=self.DINO_COLOR, linestyle="--", linewidth=1.5, alpha=0.85)
        axis.axhline(f_best, color=self.FASTER_COLOR, linestyle="--", linewidth=1.5, alpha=0.85)

        d_text_x = d_best_x + 0.45 if d_best_x < max(epochs) - 1 else d_best_x - 2.2
        d_text_y = min(0.95, d_best + 0.06) if d_best <= 0.9 else max(0.05, d_best - 0.08)
        axis.annotate(
            f"Best DINO AP@0.5: {d_best:.3f}",
            xy=(d_best_x, d_best),
            xycoords="data",
            xytext=(d_text_x, d_text_y),
            textcoords="data",
            ha="left" if d_text_x >= d_best_x else "right",
            va="bottom" if d_text_y >= d_best else "top",
            color=self.DINO_COLOR,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=self.DINO_COLOR, linewidth=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=self.DINO_COLOR, lw=1.0, alpha=0.9),
            annotation_clip=True,
        )

        f_text_x = f_best_x + 0.45 if f_best_x < max(epochs) - 1 else f_best_x - 2.2
        f_text_y = max(0.05, f_best - 0.08 if f_best > 0.88 else f_best - 0.06)
        axis.annotate(
            f"Best Faster R-CNN AP@0.5: {f_best:.3f}",
            xy=(f_best_x, f_best),
            xycoords="data",
            xytext=(f_text_x, f_text_y),
            textcoords="data",
            ha="left" if f_text_x >= f_best_x else "right",
            va="top",
            color=self.FASTER_COLOR,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=self.FASTER_COLOR, linewidth=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=self.FASTER_COLOR, lw=1.0, alpha=0.9),
            annotation_clip=True,
        )

        axis.set_title("mAP@0.5 over Epochs (Single Axis)")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("mAP@0.5")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="lower right")
        fig.tight_layout()

        output_path = self.audit.output_dir / "map_single_axis_comparison.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output_path

@torch.inference_mode()
def main():
    args = parse_args()
    config = VisualiseConfig(
        model=args.model,
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        checkpoint=args.checkpoint,
        dino_checkpoint=args.dino_checkpoint,
        fasterrcnn_checkpoint=args.fasterrcnn_checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=args.image_size,
    )
    VisualisationApp(config).run()


if __name__ == "__main__":
    main()
