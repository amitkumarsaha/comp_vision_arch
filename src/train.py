from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from src.core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    except ModuleNotFoundError:
        from src.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    from src.engine import evaluate_model, train_one_epoch
    from src.utils import count_trainable_parameters, save_json, seed_everything, utc_timestamp
else:
    try:
        from .core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    except ModuleNotFoundError:
        from .runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    from .engine import evaluate_model, train_one_epoch
    from .utils import count_trainable_parameters, save_json, seed_everything, utc_timestamp


@dataclass(frozen=True)
class TrainConfig:
    model: str
    train_data_root: str
    test_data_root: str
    output_dir: str
    subset_size: int | None
    epochs: int
    batch_size: int
    workers: int
    image_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    freeze_fasterrcnn_backbone: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Assignment 2 detector.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--train-data-root", type=str, default="data/train-validation-data")
    parser.add_argument("--test-data-root", type=str, default="data/test-data")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-fasterrcnn-backbone", action="store_true")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_dt(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class TrainingApp:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        seed_everything(config.seed)
        self.runtime = RuntimeEnvironment()
        self.data = DataPipelineManager(
            DataConfig(
                train_data_root=config.train_data_root,
                test_data_root=config.test_data_root,
                image_size=config.image_size,
                batch_size=config.batch_size,
                workers=config.workers,
                subset_size=config.subset_size,
                seed=config.seed,
            )
        )
        self.audit = AuditLogger(config.output_dir)
        self.model = ModelFactory.build(
            config.model,
            image_size=config.image_size,
            freeze_fasterrcnn_backbone=config.freeze_fasterrcnn_backbone,
        ).to(self.runtime.device)
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(self.runtime.device.type, enabled=self.runtime.device.type == "cuda")

    def run(self) -> None:
        train_dataset = self.data.train_dataset()
        test_dataset = self.data.test_dataset()
        train_loader = self.data.train_loader()
        test_loader = self.data.test_loader()

        self.audit.write_runtime("run_manifest.json", self.config, self.model, self.runtime.device)
        self.audit.write_dataset(
            "dataset_manifest.json",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_root=self.config.train_data_root,
            test_root=self.config.test_data_root,
            subset_size=self.config.subset_size,
            seed=self.config.seed,
        )

        best_map = -1.0
        history = []
        epoch_durations = []
        run_start = time.time()
        tqdm.write(f"Training started: {format_dt(run_start)}")
        tqdm.write(
            f"Config: model={self.config.model}, device={self.runtime.device}, train_images={len(train_dataset)}, "
            f"test_images={len(test_dataset)}, epochs={self.config.epochs}, batch_size={self.config.batch_size}"
        )

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            tqdm.write(f"Epoch {epoch}/{self.config.epochs} started: {format_dt(epoch_start)}")
            train_metrics = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                device=self.runtime.device,
                scaler=self.scaler,
                desc=f"train {epoch}/{self.config.epochs}",
            )
            eval_metrics = evaluate_model(
                model=self.model,
                loader=test_loader,
                device=self.runtime.device,
                desc=f"eval {epoch}/{self.config.epochs}",
            )
            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            best_map = self._record_epoch(epoch, train_metrics, eval_metrics, best_map, history, epoch_durations)

        self._write_summary(train_dataset, test_dataset, history, best_map, run_start)

    def _record_epoch(self, epoch: int, train_metrics: dict, eval_metrics: dict, best_map: float, history: list, epoch_durations: list[float]) -> float:
        avg_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = self.config.epochs - epoch
        eta_timestamp = time.time() + (avg_epoch_seconds * remaining_epochs)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        tqdm.write(
            f"Epoch {epoch}/{self.config.epochs} done in {format_duration(epoch_durations[-1])} | "
            f"loss={train_metrics.get('loss', 0.0):.4f} | "
            f"mAP@0.5={eval_metrics.get('mAP@0.5', 0.0):.4f} | "
            f"ETA completion: {format_dt(eta_timestamp)}"
        )
        self.audit.write_record(
            "training_progress.json",
            {
                "timestamp_utc": utc_timestamp(),
                "model": self.config.model,
                "history": history,
                "best_map_50": best_map,
                "epoch_durations_seconds": epoch_durations,
                "estimated_completion_local": format_dt(eta_timestamp),
            },
        )

        if eval_metrics["mAP@0.5"] > best_map:
            best_map = eval_metrics["mAP@0.5"]
            checkpoint_path = self.audit.output_dir / "best.pt"
            torch.save(
                {
                    "model_name": self.config.model,
                    "args": asdict(self.config),
                    "head_version": getattr(self.model, "checkpoint_version", "unknown"),
                    "state_dict": self.model.state_dict(),
                    "eval_metrics": eval_metrics,
                },
                checkpoint_path,
            )
            self.audit.write_record(
                "best_checkpoint.json",
                {
                    "timestamp_utc": utc_timestamp(),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "epoch": epoch,
                    "eval_metrics": eval_metrics,
                },
            )
        return best_map

    def _write_summary(self, train_dataset, test_dataset, history: list, best_map: float, run_start: float) -> None:
        summary = {
            "model": self.config.model,
            "train_images": len(train_dataset),
            "test_images": len(test_dataset),
            "trainable_parameters": count_trainable_parameters(self.model),
            "best_map_50": best_map,
            "history": history,
        }
        summary_path = self.audit.output_dir / "summary.json"
        save_json(summary, summary_path)
        self.audit.write_record(
            "training_summary.json",
            {
                "timestamp_utc": utc_timestamp(),
                "summary_path": str(summary_path.resolve()),
                "summary": summary,
            },
        )
        run_end = time.time()
        tqdm.write(
            f"Training finished: {format_dt(run_end)} | total elapsed: {format_duration(run_end - run_start)} | "
            f"best mAP@0.5={best_map:.4f}"
        )


def main():
    args = parse_args()
    config = TrainConfig(
        model=args.model,
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        freeze_fasterrcnn_backbone=args.freeze_fasterrcnn_backbone,
    )
    TrainingApp(config).run()


if __name__ == "__main__":
    main()
