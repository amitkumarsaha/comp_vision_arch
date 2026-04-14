from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppDefaults:
    train_data_root: str = str(Path("data") / "train-validation-data")
    test_data_root: str = str(Path("data") / "test-data")
    image_size: str = "320"
    batch_size: str = "4"
    workers: str = "2"
    subset_size: str = "1000"
    epochs: str = "10"
    learning_rate: str = "1e-4"
    weight_decay: str = "1e-4"
    seed: str = "42"
    dino_output: str = str(Path("outputs") / "dino")
    fasterrcnn_output: str = str(Path("outputs") / "fasterrcnn")
    viz_output: str = str(Path("outputs") / "viz" / "comparison")


class PromptSession:
    def ask(self, label: str, default: str) -> str:
        value = input(f"{label} [{default}]: ").strip()
        return value or default

    def select_train_models(self) -> list[str]:
        choice = input("Select model(s) to train: [1] DINO, [2] Faster R-CNN, [3] Both: ").strip()
        if choice == "1":
            return ["dino"]
        if choice == "2":
            return ["fasterrcnn"]
        return ["dino", "fasterrcnn"]


class CommandRunner:
    def __init__(self, root: Path, python_executable: str) -> None:
        self.root = root
        self.python_executable = python_executable

    def run_module(self, module_name: str, arguments: list[str]) -> None:
        command = [self.python_executable, "-m", module_name, *arguments]
        print("\nRunning:")
        print(" ".join(command))
        subprocess.run(command, cwd=self.root, check=True)


class TablePrinter:
    def __init__(self, widths: tuple[int, int, int] = (24, 38, 38)) -> None:
        self.widths = widths

    def _wrap_cell(self, value: str, width: int) -> list[str]:
        text = str(value) if value is not None else ""
        return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]

    def _format_row(self, row: tuple[str, str, str]) -> list[str]:
        wrapped = [self._wrap_cell(cell, width) for cell, width in zip(row, self.widths)]
        height = max(len(cell_lines) for cell_lines in wrapped)
        lines = []
        for line_index in range(height):
            parts = []
            for cell_lines, width in zip(wrapped, self.widths):
                part = cell_lines[line_index] if line_index < len(cell_lines) else ""
                parts.append(part.ljust(width))
            lines.append(" | ".join(parts))
        return lines

    def print_table(self, title: str, rows: list[tuple[str, str, str]]) -> None:
        print(f"\n{title}")
        if not rows:
            print("(no data)")
            return

        separator = "-+-".join("-" * width for width in self.widths)
        for line in self._format_row(("Field", "DINO", "Faster R-CNN")):
            print(line)
        print(separator)
        for row in rows:
            for line in self._format_row(row):
                print(line)
