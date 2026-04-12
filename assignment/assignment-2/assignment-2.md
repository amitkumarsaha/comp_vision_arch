# Assignment 2 Object Detection with a Pretrained Backbone (DINO + Head)

*(Implementation + short report)*

## Aim

The goal of this assignment is to:

- Implement an object detector using a pretrained backbone (DINO, DINOv2 or DINOv3) and your own detection head.
- Compare this approach with at least one alternative strategy (e.g. a YOLO model or a ResNet-based detector).
- Reflect on strategy choices under limited compute: what you train, what you freeze, and why.

You are free to choose your framework and libraries (e.g. PyTorch, any standard detection toolkit), but you are responsible for the design and code.

## Dataset (fixed)

Use the PASCAL VOC 2007 detection dataset.

To keep the problem small and practical:

- Use only the following three classes: `{person, car, dog}`.
- Use the official train-val split for training and test split for evaluation.
- You may further subsample the training set (e.g. to ~1000 images) if needed to fit Colab / your hardware, but you must:
  - State clearly how many images you used.
  - Use the same test set for all models you compare.

Any standard VOC2007 distribution (official site, TensorFlow Datasets, Kaggle, Hugging Face, etc.) is acceptable as long as it is the original VOC2007 detection data with correct annotations.

## Models and strategies

You must implement at least two detection strategies:

- Main model (required): DINO backbone + your own detection head
  - Use a pretrained DINO or DINOv2 backbone as a frozen feature extractor.
  - Design and implement a detection head on top of DINO features.
  - The head can be RCNN-style, anchor-based, anchor-free, grid-based, etc.
  - It must output bounding boxes + class scores for VOC2007 `person` / `car` / `dog`.
  - At minimum:
    - Train only the head parameters while keeping the DINO backbone frozen.
    - Evaluate on the VOC2007 test split (restricted to the 3 classes).
- At least one comparison strategy (choose one)
  - Choose one of the following to compare against your DINO-based detector:
    - (A) Partial fine-tuning of DINO
      - Start from your DINO + head model.
      - Unfreeze a small part of the backbone (e.g. last block / last few layers) and fine-tune it together with the head.
    - (B) YOLO-style detector
      - Use an off-the-shelf YOLO model (e.g. YOLOv8-small) and fine-tune it on the same VOC2007 subset (`person` / `car` / `dog`).
      - You may use official Ultralytics tools or another YOLO implementation.
    - (C) ResNet-based detector or tiny model from scratch
      - Implement a detector using a supervised ResNet backbone (e.g. Faster R-CNN from a standard library), or
      - A very small detector without pretraining (train from scratch).

All models must be trained and evaluated on exactly the same data (same training set, same test set, same classes) to allow comparison.

## What you need to implement

You are responsible for:

- Data pipeline for VOC2007 (loading images, boxes, labels; splitting if you subsample).
- Model definition:
  - For the DINO-based detector: how you extract features from DINO and how your head uses them.
  - For the comparison model you choose.
- Training + evaluation code:
  - Losses for detection (classification + box regression).
  - A simple evaluation metric (e.g. mAP@0.5 or another standard detection metric).
- Visualisation:
  - Draw predicted bounding boxes and labels for a few test images for each model.

You may reuse libraries and public implementations (e.g. `torchvision`, `timm`, Ultralytics YOLO), but:

- You must show clearly where DINO is used as a backbone, and
- What is your own contribution (head design, training setup, etc.).

## Suggested resources (optional)

You may find the following useful as starting points (not required):

- DINO / DINOv2 official repositories (pretrained backbones and usage examples).
- PASCAL VOC documentation and dataset descriptions.
- Ultralytics YOLOv8 docs and example training scripts (if you choose the YOLO strategy).

You are expected to search and read documentation yourself and choose a reasonable implementation path.

## Report

Write a short report (about 4-6 pages) covering:

- Task and data
  - Describe the VOC2007 subset you used:
    - Classes: `person`, `car`, `dog`.
    - Number of training and test images.
    - Any subsampling you applied.
- Model designs
  - For the DINO + head model:
    - Which DINO (or DINOv2/v3) backbone you used.
    - How you extract features and how your detection head is structured.
    - Which parameters are trained vs. frozen.
  - For your comparison model:
    - What backbone/model you used.
    - What you trained or fine-tuned.
- Experiments and results
  - A table comparing at least:
    - Model name / strategy
    - Trainable parameters (approximate)
    - Detection performance (e.g. mAP@0.5 on test)
    - 2-3 example test images with predictions from each model.
- Discussion (strategic view)
  - Under limited compute and limited labeled data:
    - Would you prefer "frozen DINO + small head" or your comparison strategy? Why?
    - If you had more data and more compute, how might your choice change?
    - How does what you observed in practice connect back to Assignment 1 ideas about:
      - Backbones vs heads
      - Supervised vs self-supervised pretraining
      - Reusing a single backbone for many tasks

## Assessment

- Correct implementation and training of DINO + head detector: 40%
- Implementation and use of at least one comparison strategy: 25%
- Quality and clarity of results and visualisations: 15%
- Discussion and strategic reasoning in the report: 20%
