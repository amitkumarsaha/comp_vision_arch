from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.audit import build_dataset_audit, build_runtime_audit, write_audit_record
    from src.data import build_data_pipeline
    from src.engine import evaluate_model
    from src.models.dino_detector import DinoGridDetector
    from src.models.faster_rcnn import build_faster_rcnn
    from src.utils import ensure_dir, utc_timestamp
else:
    from .audit import build_dataset_audit, build_runtime_audit, write_audit_record
    from .data import build_data_pipeline
    from .engine import evaluate_model
    from .models.dino_detector import DinoGridDetector
    from .models.faster_rcnn import build_faster_rcnn
    from .utils import ensure_dir, utc_timestamp


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
    output_dir = ensure_dir(Path(args.checkpoint).resolve().parent)

    pipeline = build_data_pipeline(
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=None,
        seed=42,
    )
    test_dataset = pipeline.test_dataset()
    test_loader = pipeline.test_loader()
    metrics = evaluate_model(model, test_loader, device)
    write_audit_record(output_dir, "evaluation_run_manifest.json", build_runtime_audit(args, model, device))
    write_audit_record(
        output_dir,
        "evaluation_dataset_manifest.json",
        build_dataset_audit(
            test_dataset=test_dataset,
            test_root=args.test_data_root,
            subset_size=None,
            seed=42,
        ),
    )
    write_audit_record(
        output_dir,
        "evaluation_report.json",
        {
            "timestamp_utc": utc_timestamp(),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "metrics": metrics,
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
