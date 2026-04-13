from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from src.core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    except ModuleNotFoundError:
        from src.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    from src.engine import evaluate_losses, evaluate_model
    from src.utils import utc_timestamp
else:
    try:
        from .core.runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    except ModuleNotFoundError:
        from .runtime import AuditLogger, DataConfig, DataPipelineManager, ModelFactory, RuntimeEnvironment
    from .engine import evaluate_losses, evaluate_model
    from .utils import utc_timestamp


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
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


class EvaluationApp:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.runtime = RuntimeEnvironment()
        self.model, _checkpoint = ModelFactory.load_checkpoint(
            config.model,
            checkpoint_path=config.checkpoint,
            image_size=config.image_size,
            device=self.runtime.device,
        )
        self.audit = AuditLogger(Path(config.checkpoint).resolve().parent)
        self.data = DataPipelineManager(
            DataConfig(
                train_data_root=config.train_data_root,
                test_data_root=config.test_data_root,
                image_size=config.image_size,
                batch_size=config.batch_size,
                workers=config.workers,
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
                "checkpoint_path": str(Path(self.config.checkpoint).resolve()),
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
