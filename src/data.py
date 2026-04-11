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
        self.dataset = VOCDetection(
            root=str(self.root),
            year="2007",
            image_set=image_set,
            download=download,
        )
        self.valid_indices = self._build_index()

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
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    subset_size: int | None,
    seed: int,
):
    train_dataset = VOCThreeClassDetection(
        root=data_root,
        image_set="trainval",
        image_size=image_size,
        download=True,
    )
    test_dataset = VOCThreeClassDetection(
        root=data_root,
        image_set="test",
        image_size=image_size,
        download=True,
    )

    train_dataset = build_subset(train_dataset, subset_size=subset_size, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_detection_batch,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_detection_batch,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader
