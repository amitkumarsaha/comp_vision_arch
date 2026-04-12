from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.audit import build_dataset_audit, build_runtime_audit, write_audit_record
    from src.data import build_data_pipeline
    from src.engine import evaluate_model, train_one_epoch
    from src.models.dino_detector import DinoGridDetector
    from src.models.faster_rcnn import build_faster_rcnn
    from src.utils import count_trainable_parameters, ensure_dir, save_json, seed_everything, utc_timestamp
else:
    from .audit import build_dataset_audit, build_runtime_audit, write_audit_record
    from .data import build_data_pipeline
    from .engine import evaluate_model, train_one_epoch
    from .models.dino_detector import DinoGridDetector
    from .models.faster_rcnn import build_faster_rcnn
    from .utils import count_trainable_parameters, ensure_dir, save_json, seed_everything, utc_timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Assignment 2 detector.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--train-data-root", type=str, default="data/train-validation-data")
    parser.add_argument("--test-data-root", type=str, default="data/test-data")
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


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_dt(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = build_data_pipeline(
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    train_dataset = pipeline.train_dataset()
    test_dataset = pipeline.test_dataset()
    train_loader = pipeline.train_loader()
    test_loader = pipeline.test_loader()

    model = build_model(args).to(device)
    write_audit_record(output_dir, "run_manifest.json", build_runtime_audit(args, model, device))
    write_audit_record(
        output_dir,
        "dataset_manifest.json",
        build_dataset_audit(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_root=args.train_data_root,
            test_root=args.test_data_root,
            subset_size=args.subset_size,
            seed=args.seed,
        ),
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_map = -1.0
    history = []
    epoch_durations = []
    run_start = time.time()
    tqdm.write(f"Training started: {format_dt(run_start)}")
    tqdm.write(
        f"Config: model={args.model}, device={device}, train_images={len(train_dataset)}, "
        f"test_images={len(test_dataset)}, epochs={args.epochs}, batch_size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        tqdm.write(f"Epoch {epoch}/{args.epochs} started: {format_dt(epoch_start)}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            desc=f"train {epoch}/{args.epochs}",
        )
        eval_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            desc=f"eval {epoch}/{args.epochs}",
        )
        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = args.epochs - epoch
        eta_timestamp = time.time() + (avg_epoch_seconds * remaining_epochs)
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        tqdm.write(
            f"Epoch {epoch}/{args.epochs} done in {format_duration(epoch_duration)} | "
            f"loss={train_metrics.get('loss', 0.0):.4f} | "
            f"mAP@0.5={eval_metrics.get('mAP@0.5', 0.0):.4f} | "
            f"ETA completion: {format_dt(eta_timestamp)}"
        )
        write_audit_record(
            output_dir,
            "training_progress.json",
            {
                "timestamp_utc": utc_timestamp(),
                "model": args.model,
                "history": history,
                "best_map_50": best_map,
                "epoch_durations_seconds": epoch_durations,
                "estimated_completion_local": format_dt(eta_timestamp),
            },
        )

        if eval_metrics["mAP@0.5"] > best_map:
            best_map = eval_metrics["mAP@0.5"]
            checkpoint_path = output_dir / "best.pt"
            torch.save(
                {
                    "model_name": args.model,
                    "args": vars(args),
                    "state_dict": model.state_dict(),
                    "eval_metrics": eval_metrics,
                },
                checkpoint_path,
            )
            write_audit_record(
                output_dir,
                "best_checkpoint.json",
                {
                    "timestamp_utc": utc_timestamp(),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "epoch": epoch,
                    "eval_metrics": eval_metrics,
                },
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
    write_audit_record(
        output_dir,
        "training_summary.json",
        {
            "timestamp_utc": utc_timestamp(),
            "summary_path": str((output_dir / "summary.json").resolve()),
            "summary": summary,
        },
    )
    run_end = time.time()
    tqdm.write(
        f"Training finished: {format_dt(run_end)} | total elapsed: {format_duration(run_end - run_start)} | "
        f"best mAP@0.5={best_map:.4f}"
    )


if __name__ == "__main__":
    main()
