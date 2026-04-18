from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
from src.engine import evaluate_losses, evaluate_model
from src.utils import project_path, utc_timestamp


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
        metrics = evaluate_model(self.model, test_loader, self.runtime.device, desc=f"{self.config.model} eval")

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
