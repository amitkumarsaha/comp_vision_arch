from __future__ import annotations

from pathlib import Path

from .console import AppDefaults, CommandRunner, PromptSession, TablePrinter
from .reporting import ReportRepository


class MenuApp:
    def __init__(self, root: Path, python_executable: str) -> None:
        self.root = root
        self.defaults = AppDefaults()
        self.prompts = PromptSession()
        self.runner = CommandRunner(root, python_executable)
        self.reports = ReportRepository(root)
        self.tables = TablePrinter()

    def run(self) -> None:
        while True:
            choice = self.prompt_menu()
            if choice == "1":
                self.view_reports()
            elif choice == "2":
                self.visualise_test()
            elif choice == "3":
                self.evaluate_training()
            elif choice == "4":
                self.train_models()
            elif choice == "0":
                print("Exiting.")
                break
            else:
                print("Invalid option. Please choose 0, 1, 2, 3, or 4.")

    def prompt_menu(self) -> str:
        print("\nObject Detection with a Pretrained Backbone (DINO + Head)")
        print("1. View Reports")
        print("2. Visualise Test")
        print("3. Evaluate Training")
        print("4. Train Models")
        print("0. Exit")
        return input("Select an option: ").strip()

    def _common_data_args(self) -> dict[str, str]:
        return {
            "train_data_root": self.prompts.ask("Train data root", self.defaults.train_data_root),
            "test_data_root": self.prompts.ask("Test data root", self.defaults.test_data_root),
            "image_size": self.prompts.ask("Image size", self.defaults.image_size),
        }

    def _loader_args(self, include_batch_size: bool = True) -> dict[str, str]:
        args = self._common_data_args()
        if include_batch_size:
            args["batch_size"] = self.prompts.ask("Batch size", self.defaults.batch_size)
        args["workers"] = self.prompts.ask("Workers", self.defaults.workers)
        return args

    def _checkpoint_path(self, model_name: str) -> str:
        if model_name == "dino":
            return str(Path(self.defaults.dino_output) / "best.pt")
        return str(Path(self.defaults.fasterrcnn_output) / "best.pt")

    @staticmethod
    def _pairs_to_cli(arguments: dict[str, str]) -> list[str]:
        cli_args: list[str] = []
        for key, value in arguments.items():
            cli_args.extend([f"--{key.replace('_', '-')}", value])
        return cli_args

    def _model_output_dir(self, model_name: str) -> str:
        return self.defaults.dino_output if model_name == "dino" else self.defaults.fasterrcnn_output

    def train_models(self) -> None:
        models = self.prompts.select_train_models()
        shared_args = self._loader_args(include_batch_size=True)
        shared_args.update(
            {
                "subset_size": self.prompts.ask("Subset size", self.defaults.subset_size),
                "epochs": self.prompts.ask("Epochs", self.defaults.epochs),
                "learning_rate": self.prompts.ask("Learning rate", self.defaults.learning_rate),
                "weight_decay": self.prompts.ask("Weight decay", self.defaults.weight_decay),
                "seed": self.prompts.ask("Seed", self.defaults.seed),
            }
        )

        for model_name in models:
            output_dir = self.prompts.ask(f"Output dir for {model_name}", self._model_output_dir(model_name))
            args = {"model": model_name, **shared_args, "output_dir": output_dir}
            self.runner.run_module("src.train", self._pairs_to_cli(args))

    def evaluate_training(self) -> None:
        shared_args = self._loader_args(include_batch_size=True)
        checkpoint_map = {
            "dino": self.prompts.ask("DINO checkpoint", self._checkpoint_path("dino")),
            "fasterrcnn": self.prompts.ask("Faster R-CNN checkpoint", self._checkpoint_path("fasterrcnn")),
        }

        for model_name, checkpoint in checkpoint_map.items():
            if not Path(checkpoint).exists():
                print(f"Skipping {model_name}: checkpoint not found at {checkpoint}")
                continue
            args = {"model": model_name, **shared_args, "checkpoint": checkpoint}
            self.runner.run_module("src.evaluate", self._pairs_to_cli(args))

    def visualise_test(self) -> None:
        shared_args = self._common_data_args()
        args = {
            "dino_checkpoint": self.prompts.ask("DINO checkpoint", self._checkpoint_path("dino")),
            "fasterrcnn_checkpoint": self.prompts.ask("Faster R-CNN checkpoint", self._checkpoint_path("fasterrcnn")),
            **shared_args,
            "output_dir": self.prompts.ask("Visualization output dir", self.defaults.viz_output),
            "num_images": self.prompts.ask("Number of images", "3"),
        }
        self.runner.run_module("src.visualise", self._pairs_to_cli(args))

    def view_reports(self) -> None:
        dino = self.reports.collect_model_report(Path(self.defaults.dino_output))
        fasterrcnn = self.reports.collect_model_report(Path(self.defaults.fasterrcnn_output))

        self.tables.print_table(
            "Model Setup",
            [
                ("Device", dino["device"], fasterrcnn["device"]),
                ("Trainable Parameters", dino["trainable_parameters"], fasterrcnn["trainable_parameters"]),
                ("Learning Rate", dino["learning_rate"], fasterrcnn["learning_rate"]),
                ("Weight Decay", dino["weight_decay"], fasterrcnn["weight_decay"]),
                ("Epochs Configured", dino["epochs_configured"], fasterrcnn["epochs_configured"]),
                ("Subset Size", dino["subset_size"], fasterrcnn["subset_size"]),
            ],
        )

        self.tables.print_table(
            "Dataset",
            [
                ("Train Root", dino["train_root"], fasterrcnn["train_root"]),
                ("Test Root", dino["test_root"], fasterrcnn["test_root"]),
                ("Train Images", dino["train_images"], fasterrcnn["train_images"]),
                ("Test Images", dino["test_images"], fasterrcnn["test_images"]),
                ("Train SHA-256", dino["train_sha"], fasterrcnn["train_sha"]),
                ("Test SHA-256", dino["test_sha"], fasterrcnn["test_sha"]),
            ],
        )

        self.tables.print_table(
            "Training Progress",
            [
                ("Epochs Logged", dino["epochs_logged"], fasterrcnn["epochs_logged"]),
                ("Last Epoch", dino["last_epoch"], fasterrcnn["last_epoch"]),
                ("Last Train Losses", dino["last_train_losses"], fasterrcnn["last_train_losses"]),
                ("Last Eval Metrics", dino["last_eval_metrics"], fasterrcnn["last_eval_metrics"]),
                ("Best mAP@0.5", dino["best_map_50"], fasterrcnn["best_map_50"]),
            ],
        )

        self.tables.print_table(
            "Evaluation",
            [
                ("Evaluation Losses", dino["evaluation_losses"], fasterrcnn["evaluation_losses"]),
                ("Evaluation Metrics", dino["evaluation_metrics"], fasterrcnn["evaluation_metrics"]),
            ],
        )
