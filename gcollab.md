# Google Colab Guide

This guide explains the step-by-step way to train and test this project in Google Colab.

## 1. Open Colab and enable GPU

In Colab:

1. Open a new notebook.
2. Go to `Runtime` > `Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.
4. Save.

## 2. Get the project into Colab

If the repository is on GitHub:

```python
!git clone <your-repo-url>
%cd advanced-ml
```

If the project is already stored in Google Drive, mount Drive and move into the project folder instead.

## 3. Install dependencies

Run:

```python
!pip install -r requirements.txt
```

If Colab asks for a restart after install, restart the runtime and return to the project directory.

## 4. Place the VOC2007 dataset in the expected folders

The project expects this layout:

```text
data/
  train-validation-data/
    VOCdevkit/
      VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
  test-data/
    VOCdevkit/
      VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
```

```commandline
!mkdir -p data/train-validation-data
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!tar -xvf VOCtrainval_06-Nov-2007.tar -C data/train-validation-data

!mkdir -p data/test-data
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf VOCtest_06-Nov-2007.tar -C data/test-data
```

So:

- extract `VOCtrainval_06-Nov-2007.tar` into `data/train-validation-data`
- extract `VOCtest_06-Nov-2007.tar` into `data/test-data`

## 5. Verify the dataset structure

Run:

```python
!ls data/train-validation-data/VOCdevkit/VOC2007
!ls data/test-data/VOCdevkit/VOC2007
```

You should see folders such as:

- `Annotations`
- `ImageSets`
- `JPEGImages`

## 6. Optional: set a Hugging Face token

This helps avoid unauthenticated download rate limits for the DINOv2 backbone.

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

This step is optional, but recommended.

## 7. Train the DINO model

Run:

```python
!python -m src.train \
  --model dino \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/dino \
  --subset-size 1000 \
  --epochs 10 \
  --batch-size 4 \
  --workers 2 \
  --image-size 448 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --seed 42
```

This training run uses:

- VOC2007 `trainval` for training
- VOC2007 `test` for evaluation
- only the filtered classes `person`, `car`, and `dog`

Artifacts are written to:

- `outputs/dino/best.pt`
- `outputs/dino/summary.json`
- `outputs/dino/audit/`

## 8. Train the Faster R-CNN comparison model

Run:

```python
!python -m src.train \
  --model fasterrcnn \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/fasterrcnn \
  --subset-size 1000 \
  --epochs 10 \
  --batch-size 4 \
  --workers 2 \
  --image-size 448 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --seed 42
```

Artifacts are written to:

- `outputs/fasterrcnn/best.pt`
- `outputs/fasterrcnn/summary.json`
- `outputs/fasterrcnn/audit/`

Note:

- In a normal single Colab runtime, step 7 and step 8 should be run sequentially, not in parallel.
- Both are GPU-heavy jobs and will compete for the same runtime resources if started together.

## 9. Evaluate the trained DINO model

Run:

```python
!python -m src.evaluate \
  --model dino \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --checkpoint outputs/dino/best.pt \
  --image-size 448 \
  --batch-size 4 \
  --workers 2
```

This produces:

- detection losses
- evaluation metrics including `mAP@0.5`
- saved evaluation artifacts in `outputs/dino/audit/`

## 10. Evaluate the Faster R-CNN model

Run:

```python
!python -m src.evaluate \
  --model fasterrcnn \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --checkpoint outputs/fasterrcnn/best.pt \
  --image-size 448 \
  --batch-size 4 \
  --workers 2
```

This writes evaluation results into:

- `outputs/fasterrcnn/audit/`

## 11. Generate side-by-side visualisations

Run:

```python
!python -m src.visualise \
  --dino-checkpoint outputs/dino/best.pt \
  --fasterrcnn-checkpoint outputs/fasterrcnn/best.pt \
  --train-data-root data/train-validation-data \
  --test-data-root data/test-data \
  --output-dir outputs/viz/comparison \
  --num-images 3 \
  --image-size 448
```

This creates:

- per-image side-by-side comparison images
- diagnostic images with ground truth
- one combined image grid

Outputs are written to:

- `outputs/viz/comparison/`

## 12. Display the generated comparison image in Colab

Run:

```python
from IPython.display import Image, display
display(Image(filename="outputs/viz/comparison/comparison_grid.png"))
```

## 13. Use the saved artifacts for reporting

Useful files for the report:

- `outputs/dino/summary.json`
- `outputs/dino/audit/`
- `outputs/fasterrcnn/summary.json`
- `outputs/fasterrcnn/audit/`
- `outputs/viz/comparison/`

These contain:

- train and test image counts
- subset size used
- trainable parameter counts
- losses
- metrics
- dataset SHA-256 fingerprints
- generated visualisations

## 14. Save outputs to Google Drive

Colab runtimes are temporary, so save the outputs if you want to keep the checkpoints and results.

Example:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r outputs /content/drive/MyDrive/advanced-ml-outputs
```

## Recommended execution order

1. Install dependencies.
2. Set up the dataset folders.
3. Train `dino`.
4. Train `fasterrcnn`.
5. Evaluate both models.
6. Generate visualisations.
7. Save `outputs/` and use the audit files for reporting.
