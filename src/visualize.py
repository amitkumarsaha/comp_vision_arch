from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from .data import build_dataloaders
from .models.dino_detector import DinoGridDetector
from .models.faster_rcnn import build_faster_rcnn
from .utils import IDX_TO_CLASS, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Render predictions for a few test images.")
    parser.add_argument("--model", choices=["dino", "fasterrcnn"], required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args()


def build_model(model_name: str, image_size: int):
    if model_name == "dino":
        return DinoGridDetector(image_size=image_size)
    return build_faster_rcnn()


def draw_predictions(image: torch.Tensor, prediction: dict[str, torch.Tensor], output_path):
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image.permute(1, 2, 0).cpu().numpy())
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        x1, y1, x2, y2 = box.tolist()
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        axis.text(x1, y1, f"{IDX_TO_CLASS[int(label)]}: {float(score):.2f}", color="white", backgroundcolor="black")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


@torch.inference_mode()
def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.image_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    _train_dataset, _test_dataset, _train_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=1,
        workers=0,
        subset_size=None,
        seed=42,
    )

    saved = 0
    for images, targets, meta in test_loader:
        images = [image.to(device) for image in images]
        outputs = model(images, targets)
        predictions = outputs if isinstance(outputs, list) else outputs.predictions
        prediction = {key: value.detach().cpu() for key, value in predictions[0].items()}
        draw_predictions(images[0].cpu(), prediction, output_dir / f"{meta[0].image_id}.png")
        saved += 1
        if saved >= args.num_images:
            break


if __name__ == "__main__":
    main()
