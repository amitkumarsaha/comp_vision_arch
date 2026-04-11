from __future__ import annotations

import argparse

import torch

from .data import build_dataloaders
from .engine import evaluate_model, train_one_epoch
from .models.dino_detector import DinoGridDetector
from .models.faster_rcnn import build_faster_rcnn
from .utils import count_trainable_parameters, ensure_dir, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Assignment 2 detector.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--data-root", type=str, default="data")
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


def build_model(args):
    if args.model == "dino":
        return DinoGridDetector(image_size=args.image_size)
    return build_faster_rcnn(train_backbone=not args.freeze_fasterrcnn_backbone)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_map = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, scaler=scaler)
        eval_metrics = evaluate_model(model=model, loader=test_loader, device=device)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        print(row)

        if eval_metrics["mAP@0.5"] > best_map:
            best_map = eval_metrics["mAP@0.5"]
            torch.save(
                {
                    "model_name": args.model,
                    "args": vars(args),
                    "state_dict": model.state_dict(),
                    "eval_metrics": eval_metrics,
                },
                output_dir / "best.pt",
            )

    summary = {
        "model": args.model,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "trainable_parameters": count_trainable_parameters(model),
        "best_map_50": best_map,
        "history": history,
    }
    save_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
