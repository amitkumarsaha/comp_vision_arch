from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

VOC_CLASSES = ["person", "car", "dog"]
CLASS_TO_IDX = {name: idx + 1 for idx, name in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def collate_detection_batch(batch):
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    meta = [item["meta"] for item in batch]
    return images, targets, meta
