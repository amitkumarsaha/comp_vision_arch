from __future__ import annotations

import platform
from pathlib import Path

import torch
from torch.utils.data import Subset

from src.utils import CLASS_TO_IDX, VOC_CLASSES, ensure_dir, save_json, sha256_text, project_path, utc_timestamp


def build_runtime_audit(args, model, device: torch.device) -> dict:
    return {
        "timestamp_utc": utc_timestamp(),
        "model": args.model,
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "config": vars(args),
        "classes": {
            "names": VOC_CLASSES,
            "mapping": CLASS_TO_IDX,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _unwrap_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, dataset.indices
    return dataset, list(range(len(dataset)))


def dataset_image_ids(dataset) -> list[str]:
    base_dataset, indices = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "image_id_at"):
        return [base_dataset.image_id_at(index) for index in indices]

    image_ids = []
    if hasattr(base_dataset, "valid_indices") and hasattr(base_dataset, "dataset"):
        for subset_index in indices:
            dataset_index = base_dataset.valid_indices[subset_index]
            target = base_dataset.dataset[dataset_index][1]
            image_ids.append(target["annotation"]["filename"])
        return image_ids

    for index in indices:
        sample = base_dataset[index]
        target = sample["target"] if isinstance(sample, dict) else sample[1]
        image_id = target.get("image_id", f"index-{index}")
        image_ids.append(str(image_id))
    return image_ids


def build_split_audit(dataset, root: str | Path, split_name: str) -> dict:
    image_ids = dataset_image_ids(dataset)
    return {
        "split": split_name,
        "root": project_path(root),
        "image_count": len(image_ids),
        "image_ids": image_ids,
        "sha256": sha256_text(image_ids),
    }


def build_dataset_audit(
    train_dataset=None,
    test_dataset=None,
    train_root: str | Path | None = None,
    test_root: str | Path | None = None,
    subset_size: int | None = None,
    seed: int | None = None,
) -> dict:
    payload = {
        "timestamp_utc": utc_timestamp(),
        "subset_size_requested": subset_size,
        "subset_seed": seed,
    }
    if train_dataset is not None and train_root is not None:
        payload["train"] = build_split_audit(train_dataset, train_root, "trainval")
    if test_dataset is not None and test_root is not None:
        payload["test"] = build_split_audit(test_dataset, test_root, "test")
    return payload


def init_audit_dir(output_dir: str | Path) -> Path:
    return ensure_dir(Path(output_dir) / "audit")


def write_audit_record(output_dir: str | Path, name: str, payload: dict) -> Path:
    audit_dir = init_audit_dir(output_dir)
    path = audit_dir / name
    save_json(payload, path)
    return path
