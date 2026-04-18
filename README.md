# Object Detection with a Pretrained Backbone

This project implements the Assignment 2 pipeline for object detection on **PASCAL VOC 2007** using the class subset `{person, car, dog}`.

It includes:

- A **frozen DINOv2 backbone** with a **custom grid-based detection head**
- A **ResNet-50 Faster R-CNN baseline** for comparison
- VOC2007 loading, filtering, training, evaluation, and visualisation scripts
- A short report draft in `report/assignment-2/assignment2_report.md`

For the Assignment 2 brief, this maps to:

- Required main strategy: **DINOv2 backbone + your own detection head**
- Comparison strategy (option C): **ResNet-based detector via Faster R-CNN**

## Setup

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you are working on this repository on macOS with restricted cache permissions, create local cache directories first:

```bash
mkdir -p .cache/torch/hub/checkpoints .cache/huggingface .cache/matplotlib .cache/fontconfig logs
```

## Interactive Menu

You can launch the project menu from the repository root:

```bash
python -m main
```

The menu provides:

- `1. View Reports`: displays saved model setup, dataset, training, and evaluation details from the audit trail
- `2. Visualise Test`: runs the existing visualization flow, including DINO vs Faster R-CNN comparison images
- `3. Evaluate Training`: evaluates both models and reports average detection losses plus `mAP@0.5`
- `4. Train Models`: prompts for model selection and training parameters, using defaults when left blank

## Dataset Layout

The training scripts expect:

- `data/train-validation-data/VOC2007/...`
- `data/test-data/VOC2007/...`

The loader also accepts a flat `VOC2007` folder and will resolve it automatically.

## Working Training Configuration

The successful final runs used:

- Classes: `person`, `car`, `dog`
- Train subset: `1000` images
- Test split: filtered VOC2007 test set (`2895` images)
- Seed: `42`
- Image size: `448`
- Batch size: `4`
- Epochs: `10`
- Workers: `0`

`--workers 0` was necessary on this machine because multiprocessing workers crashed with shared-memory permission errors.

## Train the DINO detector

Use local caches plus offline flags once `facebook/dinov2-small` has been downloaded or cached once on the machine:

```bash
TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$HOME/.cache/huggingface" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m src.train \
  --model dino \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/dino-final \
  --subset-size 1000 \
  --epochs 10 \
  --batch-size 4 \
  --image-size 448 \
  --workers 0
```

## Train the Faster R-CNN baseline

```bash
TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$PWD/.cache/huggingface" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
python -m src.train \
  --model fasterrcnn \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/fasterrcnn-final \
  --subset-size 1000 \
  --epochs 10 \
  --batch-size 4 \
  --image-size 448 \
  --workers 0
```

## Evaluate a checkpoint

The evaluation script now reads `image_size` from checkpoint metadata automatically, so you do not need to pass `--image-size` when the checkpoint was trained at `448`.

```bash
TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$HOME/.cache/huggingface" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m src.evaluate \
  --model dino \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --checkpoint outputs/dino-final/best.pt
```

## Export qualitative predictions

```bash
TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$HOME/.cache/huggingface" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m src.visualise \
  --model dino \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --checkpoint outputs/dino-final/best.pt \
  --output-dir outputs/viz/dino-final \
  --num-images 3
```

## Export side-by-side comparison figures

```bash
TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$HOME/.cache/huggingface" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m src.visualise \
  --dino-checkpoint outputs/dino-final/best.pt \
  --fasterrcnn-checkpoint outputs/fasterrcnn-final/best.pt \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/viz/comparison-final \
  --num-images 3
```

## Notes

- The DINO-based model keeps the pretrained backbone frozen by default.
- The training split is loaded from `data/train-validation-data` and the official VOC2007 test split from `data/test-data`.
- The Faster R-CNN baseline fine-tunes a supervised detector on the exact same train subset and test split.
- All metrics are computed on the **same test split** and **same three classes**.
- On this Apple Silicon machine, PyTorch MPS was unavailable in practice, so all successful final runs were CPU-based.
- If your hardware is limited, keep `--subset-size` around `500-1000`.
- If `transformers` tries to download DINO again during evaluation or visualization, point `HF_HOME` at the existing Hugging Face cache and use offline flags.

## Audit Trail

Each training, evaluation, and visualization run writes machine-readable audit records so the experiment can be verified later.

- Training writes `audit/run_manifest.json`, `audit/dataset_manifest.json`, `audit/training_progress.json`, `audit/best_checkpoint.json`, and `audit/training_summary.json` inside the model output directory.
- Evaluation writes `audit/evaluation_run_manifest.json`, `audit/evaluation_dataset_manifest.json`, and `audit/evaluation_report.json` next to the checkpoint output directory.
- Visualization writes `audit/visualization_run_manifest.json`, `audit/visualization_dataset_manifest.json`, and `audit/visualization_manifest.json` inside the visualization output directory.

The dataset manifests include the exact filtered image ids and SHA-256 digests for the train and test sets, which makes it easy to verify that the same test set was used across models.
