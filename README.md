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

## Train the DINO detector

```powershell
python -m src.assignment2.train `
  --model dino `
  --data-root .\data `
  --output-dir .\outputs\dino `
  --subset-size 1000 `
  --epochs 10 `
  --batch-size 4
```

## Train the Faster R-CNN baseline

```powershell
python -m src.assignment2.train `
  --model fasterrcnn `
  --data-root .\data `
  --output-dir .\outputs\fasterrcnn `
  --subset-size 1000 `
  --epochs 8 `
  --batch-size 2
```

## Evaluate a checkpoint

```powershell
python -m src.assignment2.evaluate `
  --model dino `
  --data-root .\data `
  --checkpoint .\outputs\dino\best.pt
```

## Export qualitative predictions

```powershell
python -m src.assignment2.visualize `
  --model fasterrcnn `
  --data-root .\data `
  --checkpoint .\outputs\fasterrcnn\best.pt `
  --output-dir .\outputs\viz\fasterrcnn `
  --num-images 3
```

## Notes

- The DINO-based model keeps the pretrained backbone frozen by default.
- The Faster R-CNN baseline fine-tunes a supervised detector on the exact same train subset and test split.
- All metrics are computed on the **same test split** and **same three classes**.
- If your hardware is limited, keep `--subset-size` around `500-1000` and use mixed precision when CUDA is available.
