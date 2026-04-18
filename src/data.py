from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
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
        self.voc2007_root = self._resolve_voc2007_root(self.root)
        self.image_set = image_set
        self.records = self._build_records()

    @staticmethod
    def _resolve_voc2007_root(root: Path) -> Path:
        if (root / "VOC2007").exists():
            return root / "VOC2007"
        if root.name == "VOC2007" and root.exists():
            return root
        raise RuntimeError(
            f"VOC2007 data not found under {root}. Expected {root / 'VOC2007'} or direct VOC2007 path."
        )

    def _split_file(self) -> Path:
        return self.voc2007_root / "ImageSets" / "Main" / f"{self.image_set}.txt"

    def _parse_annotation(self, image_id: str):
        annotation_path = self.voc2007_root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(annotation_path).getroot()
        boxes = []
        labels = []
        difficult = []

        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if name not in CLASS_TO_IDX:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", default="0")) - 1.0
            ymin = float(bndbox.findtext("ymin", default="0")) - 1.0
            xmax = float(bndbox.findtext("xmax", default="0")) - 1.0
            ymax = float(bndbox.findtext("ymax", default="0")) - 1.0
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])
            difficult.append(int(obj.findtext("difficult", default="0")))

        return boxes, labels, difficult

    def _build_records(self) -> list[dict]:
        split_file = self._split_file()
        if not split_file.exists():
            raise RuntimeError(f"Missing VOC split file: {split_file}")

        image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = []
        for image_id in image_ids:
            boxes, labels, difficult = self._parse_annotation(image_id)
            if not boxes:
                continue
            records.append(
                {
                    "image_id": image_id,
                    "filename": f"{image_id}.jpg",
                    "image_path": self.voc2007_root / "JPEGImages" / f"{image_id}.jpg",
                    "boxes": boxes,
                    "labels": labels,
                    "difficult": difficult,
                }
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def image_id_at(self, index: int) -> str:
        return self.records[index]["filename"]

    def __getitem__(self, index: int):
        record = self.records[index]
        pil_image = Image.open(record["image_path"]).convert("RGB")

        orig_w, orig_h = pil_image.size
        resized = F.resize(pil_image, [self.image_size, self.image_size])
        image_tensor = F.to_tensor(resized)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        resized_boxes = torch.tensor(record["boxes"], dtype=torch.float32)
        resized_boxes[:, [0, 2]] *= scale_x
        resized_boxes[:, [1, 3]] *= scale_y

        target_dict = {
            "boxes": resized_boxes,
            "labels": torch.tensor(record["labels"], dtype=torch.long),
            "difficult": torch.tensor(record["difficult"], dtype=torch.long),
            "image_id": record["filename"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        meta = SampleMeta(
            image_id=record["filename"],
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
