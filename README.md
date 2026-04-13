# Object Detection with a Pretrained Backbone

This project implements the Assignment 2 pipeline for object detection on **PASCAL VOC 2007** using the class subset `{person, car, dog}`.

It includes:

- A **frozen DINOv2 backbone** with a **custom grid-based detection head**
- A **ResNet-50 Faster R-CNN baseline** for comparison
- VOC2007 loading, filtering, training, evaluation, and visualisation scripts
- A short report draft in [`report/assignment2_report.md`](C:/Users/Amit/Projects/github/advanced-ml/report/assignment2_report.md)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Interactive Menu

You can launch the project menu from the repository root:

```powershell
python .\main.py
```

The menu provides:

- `1. Visualise`: runs the existing visualization flow, including DINO vs Faster R-CNN comparison images
- `2. Evaluate`: evaluates both models and reports average detection losses plus `mAP@0.5`
- `3. Report`: displays saved model setup, dataset, training, and evaluation details from the audit trail
- `4. Train Models`: prompts for model selection and training parameters, using defaults when left blank

## Train the DINO detector

```powershell
python -m src.train `
  --model dino `
  --train-data-root .\data\train-validation-data `
  --test-data-root .\data\test-data `
  --output-dir .\outputs\dino `
  --subset-size 1000 `
  --epochs 10 `
  --batch-size 4
```

## Train the Faster R-CNN baseline

```powershell
python -m src.train `
  --model fasterrcnn `
  --train-data-root .\data\train-validation-data `
  --test-data-root .\data\test-data `
  --output-dir .\outputs\fasterrcnn `
  --subset-size 1000 `
  --epochs 10 `
  --batch-size 4
```

## Evaluate a checkpoint

```powershell
python -m src.evaluate `
  --model dino `
  --train-data-root .\data\train-validation-data `
  --test-data-root .\data\test-data `
  --checkpoint .\outputs\dino\best.pt
```

## Export qualitative predictions

```powershell
python -m src.visualize `
  --model fasterrcnn `
  --train-data-root .\data\train-validation-data `
  --test-data-root .\data\test-data `
  --checkpoint .\outputs\fasterrcnn\best.pt `
  --output-dir .\outputs\viz\fasterrcnn `
  --num-images 3
```

## Export side-by-side comparison figures

```powershell
python -m src.visualize `
  --dino-checkpoint .\outputs\dino\best.pt `
  --fasterrcnn-checkpoint .\outputs\fasterrcnn\best.pt `
  --train-data-root .\data\train-validation-data `
  --test-data-root .\data\test-data `
  --output-dir .\outputs\viz\comparison `
  --num-images 3
```

## Notes

- The DINO-based model keeps the pretrained backbone frozen by default.
- The training split is loaded from `data/train-validation-data` and the official VOC2007 test split from `data/test-data`.
- The Faster R-CNN baseline fine-tunes a supervised detector on the exact same train subset and test split.
- All metrics are computed on the **same test split** and **same three classes**.
- If your hardware is limited, keep `--subset-size` around `500-1000` and use mixed precision when CUDA is available.

## Audit Trail

Each training, evaluation, and visualization run writes machine-readable audit records so the experiment can be verified later.

- Training writes `audit/run_manifest.json`, `audit/dataset_manifest.json`, `audit/training_progress.json`, `audit/best_checkpoint.json`, and `audit/training_summary.json` inside the model output directory.
- Evaluation writes `audit/evaluation_run_manifest.json`, `audit/evaluation_dataset_manifest.json`, and `audit/evaluation_report.json` next to the checkpoint output directory.
- Visualization writes `audit/visualization_run_manifest.json`, `audit/visualization_dataset_manifest.json`, and `audit/visualization_manifest.json` inside the visualization output directory.

The dataset manifests include the exact filtered image ids and SHA-256 digests for the train and test sets, which makes it easy to verify that the same test set was used across models.

If you prefer running scripts directly, these also work now:

```powershell
python .\src\train.py --model dino --train-data-root .\data\train-validation-data --test-data-root .\data\test-data --output-dir .\outputs\dino
python .\src\evaluate.py --model dino --train-data-root .\data\train-validation-data --test-data-root .\data\test-data --checkpoint .\outputs\dino\best.pt
python .\src\visualize.py --model dino --train-data-root .\data\train-validation-data --test-data-root .\data\test-data --checkpoint .\outputs\dino\best.pt --output-dir .\outputs\viz\dino
python .\src\visualize.py --dino-checkpoint .\outputs\dino\best.pt --fasterrcnn-checkpoint .\outputs\fasterrcnn\best.pt --train-data-root .\data\train-validation-data --test-data-root .\data\test-data --output-dir .\outputs\viz\comparison
```
