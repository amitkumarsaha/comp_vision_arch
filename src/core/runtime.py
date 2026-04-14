from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..audit import build_dataset_audit, build_runtime_audit, write_audit_record
from ..data import build_data_pipeline
from ..models.dino_detector import DinoGridDetector, build_dino_model_for_checkpoint
from ..models.faster_rcnn import build_faster_rcnn
from ..utils import ensure_dir


@dataclass(frozen=True)
class DataConfig:
    train_data_root: str
    test_data_root: str
    image_size: int
    batch_size: int
    workers: int
    subset_size: int | None
    seed: int


class RuntimeEnvironment:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


class DataPipelineManager:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.pipeline = build_data_pipeline(
            train_data_root=config.train_data_root,
            test_data_root=config.test_data_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            workers=config.workers,
            subset_size=config.subset_size,
            seed=config.seed,
        )

    def train_dataset(self):
        return self.pipeline.train_dataset()

    def test_dataset(self):
        return self.pipeline.test_dataset()

    def train_loader(self):
        return self.pipeline.train_loader()

    def test_loader(self, batch_size: int | None = None, workers: int | None = None):
        return self.pipeline.test_loader(batch_size=batch_size, workers=workers)


class ModelFactory:
    @staticmethod
    def build(model_name: str, image_size: int, freeze_fasterrcnn_backbone: bool = False):
        if model_name == "dino":
            return DinoGridDetector(image_size=image_size)
        return build_faster_rcnn(train_backbone=not freeze_fasterrcnn_backbone)

    @classmethod
    def load_checkpoint(cls, model_name: str, checkpoint_path: str | Path, image_size: int, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if model_name == "dino":
            model = build_dino_model_for_checkpoint(checkpoint, image_size=image_size)
        else:
            model = cls.build(model_name, image_size=image_size)
            model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model, checkpoint


class AuditLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)

    def write_runtime(self, filename: str, args, model, device: torch.device) -> None:
        write_audit_record(self.output_dir, filename, build_runtime_audit(args, model, device))

    def write_dataset(
        self,
        filename: str,
        *,
        train_dataset=None,
        test_dataset=None,
        train_root: str | None = None,
        test_root: str | None = None,
        subset_size: int | None = None,
        seed: int = 42,
    ) -> None:
        write_audit_record(
            self.output_dir,
            filename,
            build_dataset_audit(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_root=train_root,
                test_root=test_root,
                subset_size=subset_size,
                seed=seed,
            ),
        )

    def write_record(self, filename: str, payload: dict) -> None:
        write_audit_record(self.output_dir, filename, payload)
