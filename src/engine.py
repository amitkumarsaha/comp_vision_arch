from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from .metrics import mean_average_precision


def move_targets_to_device(targets, device: torch.device):
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
        )
    return moved


def _extract_loss(output):
    if isinstance(output, dict):
        return sum(output.values())
    if hasattr(output, "losses") and output.losses is not None:
        return output.losses["loss"]
    raise TypeError("Unsupported model output for loss extraction.")


def _extract_loss_dict(output):
    if isinstance(output, dict):
        return output
    if hasattr(output, "losses") and output.losses is not None:
        return output.losses
    raise TypeError("Unsupported model output for loss extraction.")


def _extract_predictions(output):
    if isinstance(output, list):
        return output
    if hasattr(output, "predictions") and output.predictions is not None:
        return output.predictions
    raise TypeError("Unsupported model output for prediction extraction.")


def train_one_epoch(model, loader, optimizer, device: torch.device, scaler=None):
    model.train()
    running = defaultdict(float)

    for images, targets, _meta in tqdm(loader, desc="train", leave=False):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type=device.type):
                output = model(images, targets)
                loss = _extract_loss(output)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images, targets)
            loss = _extract_loss(output)
            loss.backward()
            optimizer.step()

        for key, value in _extract_loss_dict(output).items():
            running[key] += float(value.detach().item())

    total_steps = max(len(loader), 1)
    return {key: value / total_steps for key, value in running.items()}


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device):
    model.eval()
    predictions = []
    targets = []

    for images, batch_targets, _meta in tqdm(loader, desc="eval", leave=False):
        images = [image.to(device) for image in images]
        moved_targets = move_targets_to_device(batch_targets, device)
        output = model(images, moved_targets)
        batch_predictions = _extract_predictions(output)

        for pred in batch_predictions:
            predictions.append({key: value.detach().cpu() for key, value in pred.items()})
        for target in batch_targets:
            targets.append({key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in target.items()})

    return mean_average_precision(predictions=predictions, targets=targets)
