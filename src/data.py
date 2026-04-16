from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F

from .utils import CLASS_TO_IDX, collate_detection_batch


@dataclass
class SampleMeta:
    image_id: str
    original_size: tuple[int, int]
    resized_size: tuple[int, int]


@dataclass
class VOCDataPipeline:
    train_data_root: str | Path
    test_data_root: str | Path
    image_size: int
    batch_size: int
    workers: int
    subset_size: int | None
    seed: int
    _train_dataset: Dataset | None = None
    _test_dataset: Dataset | None = None

    @property
    def pin_memory(self) -> bool:
        return torch.cuda.is_available()

    def train_dataset(self) -> Dataset:
        if self._train_dataset is None:
            dataset = VOCThreeClassDetection(
                root=self.train_data_root,
                image_set="trainval",
                image_size=self.image_size,
                download=False,
            )
            self._train_dataset = build_subset(dataset, subset_size=self.subset_size, seed=self.seed)
        return self._train_dataset

    def test_dataset(self) -> Dataset:
        if self._test_dataset is None:
            self._test_dataset = VOCThreeClassDetection(
                root=self.test_data_root,
                image_set="test",
                image_size=self.image_size,
                download=False,
            )
        return self._test_dataset

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=collate_detection_batch,
            pin_memory=self.pin_memory,
        )

    def test_loader(self, batch_size: int | None = None, workers: int | None = None) -> DataLoader:
        return DataLoader(
            self.test_dataset(),
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.workers if workers is None else workers,
            collate_fn=collate_detection_batch,
            pin_memory=self.pin_memory,
        )


class VOCThreeClassDetection(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_set: str,
        image_size: int = 448,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.dataset_root = self._resolve_voc_root(self.root)
        self.dataset = VOCDetection(
            root=str(self.dataset_root),
            year="2007",
            image_set=image_set,
            download=download,
        )
        self.valid_indices = self._build_index()

    @staticmethod
    def _resolve_voc_root(root: Path) -> Path:
        if (root / "VOCdevkit" / "VOC2007").exists():
            return root
        if (root / "VOC2007").exists():
            adapted_root = root / "VOCdevkit"
            adapted_root.mkdir(parents=True, exist_ok=True)
            target = adapted_root / "VOC2007"
            if not target.exists():
                target.symlink_to((root / "VOC2007").resolve())
            return root
        raise RuntimeError(
            f"VOC2007 data not found under {root}. Expected either VOCdevkit/VOC2007 or VOC2007."
        )

    def _build_index(self) -> list[int]:
        valid = []
        for idx in range(len(self.dataset)):
            _image, target = self.dataset[idx]
            objects = target["annotation"].get("object", [])
            if isinstance(objects, dict):
                objects = [objects]
            if any(obj["name"] in CLASS_TO_IDX for obj in objects):
                valid.append(idx)
        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.valid_indices[index]]
        annotation = target["annotation"]
        objects = annotation.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        boxes = []
        labels = []
        difficult = []
        for obj in objects:
            name = obj["name"]
            if name not in CLASS_TO_IDX:
                continue
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"]) - 1.0
            ymin = float(bbox["ymin"]) - 1.0
            xmax = float(bbox["xmax"]) - 1.0
            ymax = float(bbox["ymax"]) - 1.0
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])
            difficult.append(int(obj.get("difficult", 0)))

        if not boxes:
            raise RuntimeError("Filtered VOC sample unexpectedly has no retained boxes.")

        orig_w, orig_h = image.size
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])
        image_tensor = F.to_tensor(image)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        resized_boxes = torch.tensor(boxes, dtype=torch.float32)
        resized_boxes[:, [0, 2]] *= scale_x
        resized_boxes[:, [1, 3]] *= scale_y

        target_dict = {
            "boxes": resized_boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficult": torch.tensor(difficult, dtype=torch.long),
            "image_id": annotation["filename"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        meta = SampleMeta(
            image_id=annotation["filename"],
            original_size=(orig_h, orig_w),
            resized_size=(self.image_size, self.image_size),
        )
        return {"image": image_tensor, "target": target_dict, "meta": meta}


def build_subset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    train_data_root: str | Path,
    test_data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    subset_size: int | None,
    seed: int,
):
    pipeline = build_data_pipeline(
        train_data_root=train_data_root,
        test_data_root=test_data_root,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        subset_size=subset_size,
        seed=seed,
    )
    train_dataset = pipeline.train_dataset()
    test_dataset = pipeline.test_dataset()
    train_loader = pipeline.train_loader()
    test_loader = pipeline.test_loader()
    return train_dataset, test_dataset, train_loader, test_loader


def build_data_pipeline(
    train_data_root: str | Path,
    test_data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    subset_size: int | None,
    seed: int,
) -> VOCDataPipeline:
    return VOCDataPipeline(
        train_data_root=train_data_root,
        test_data_root=test_data_root,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        subset_size=subset_size,
        seed=seed,
    )
