from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportRow:
    field: str
    dino: str
    fasterrcnn: str


class ReportRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _to_relative_path(self, value: str) -> str:
        if not value or value == "n/a":
            return value
        try:
            return str(Path(value).resolve().relative_to(self.root))
        except Exception:
            return value

    def collect_model_report(self, output_dir: Path) -> dict[str, str]:
        run_manifest = self._read_json(output_dir / "audit" / "run_manifest.json") or {}
        dataset_manifest = self._read_json(output_dir / "audit" / "dataset_manifest.json") or {}
        training_progress = self._read_json(output_dir / "audit" / "training_progress.json") or {}
        training_summary = self._read_json(output_dir / "summary.json") or {}
        evaluation_report = self._read_json(output_dir / "audit" / "evaluation_report.json") or {}

        config = run_manifest.get("config", {})
        history = training_progress.get("history", [])
        last_history = history[-1] if history else {}

        return {
            "device": str(run_manifest.get("device", "n/a")),
            "trainable_parameters": str(run_manifest.get("trainable_parameters", "n/a")),
            "train_root": self._to_relative_path(str(config.get("train_data_root", "n/a"))),
            "test_root": self._to_relative_path(str(config.get("test_data_root", "n/a"))),
            "subset_size": str(config.get("subset_size", "n/a")),
            "epochs_configured": str(config.get("epochs", "n/a")),
            "learning_rate": str(config.get("learning_rate", "n/a")),
            "weight_decay": str(config.get("weight_decay", "n/a")),
            "train_images": str(dataset_manifest.get("train", {}).get("image_count", "n/a")),
            "test_images": str(dataset_manifest.get("test", {}).get("image_count", "n/a")),
            "train_sha": str(dataset_manifest.get("train", {}).get("sha256", "n/a")),
            "test_sha": str(dataset_manifest.get("test", {}).get("sha256", "n/a")),
            "epochs_logged": str(len(history)),
            "last_epoch": str(last_history.get("epoch", "n/a")),
            "last_train_losses": str(last_history.get("train", "n/a")),
            "last_eval_metrics": str(last_history.get("eval", "n/a")),
            "best_map_50": str(training_summary.get("best_map_50", "n/a")),
            "evaluation_losses": str(evaluation_report.get("losses", "n/a")),
            "evaluation_metrics": str(evaluation_report.get("metrics", "n/a")),
        }
