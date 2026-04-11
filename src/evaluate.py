from __future__ import annotations

import argparse

import torch

from .data import build_dataloaders
from .engine import evaluate_model
from .models.dino_detector import DinoGridDetector
from .models.faster_rcnn import build_faster_rcnn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Assignment 2 detector.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def build_model(model_name: str, image_size: int):
    if model_name == "dino":
        return DinoGridDetector(image_size=image_size)
    return build_faster_rcnn()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.image_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    _train_dataset, _test_dataset, _train_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=None,
        seed=42,
    )
    print(evaluate_model(model, test_loader, device))


if __name__ == "__main__":
    main()
