# Assignment 2: Object Detection with a Pretrained Backbone

## 1. Task and Data

This project studies object detection on the **PASCAL VOC 2007 detection dataset** using the restricted class set **{person, car, dog}**. The official **trainval** split is used for training and the official **test** split is used for evaluation. To keep training practical under limited compute, the training split can optionally be subsampled; in the experiments below the same subset is used for every compared model.

Fill in after running:

- Training images used: `[...]`
- Test images used after class filtering: `[...]`
- Subsampling applied: `[...]`

## 2. Model Designs

### 2.1 Frozen DINOv2 + Custom Detection Head

The main model uses a pretrained **DINOv2** visual backbone as a **frozen feature extractor**. The implementation loads a pretrained DINOv2 encoder and uses the final patch-token activations, reshaped into a spatial feature map. A lightweight custom detection head is then trained on top of those frozen features.

The custom head is a simple **grid-based detector**:

- Two convolutional layers form a small adaptation stem over the frozen DINO feature map.
- A classification branch predicts one of `{background, person, car, dog}` per spatial location.
- A regression branch predicts a normalized bounding box `(cx, cy, w, h)` per spatial location.
- Targets are assigned by the ground-truth box center: each object is mapped to the grid cell containing its center.

Training strategy:

- **Frozen parameters**: all DINO backbone weights
- **Trainable parameters**: only the custom detection head
- Losses:
  - Cross-entropy for classification
  - L1 box regression loss
  - Generalized IoU loss

### 2.2 Comparison Model: Faster R-CNN with ResNet-50 FPN

The comparison model is a **supervised ResNet-based detector**, implemented with **Faster R-CNN + ResNet-50 FPN** from `torchvision`. The detection predictor head is replaced so that the model predicts only the relevant classes plus background.

Training strategy:

- Backbone: pretrained supervised ResNet-50 FPN detector backbone
- Detection predictor: replaced and fine-tuned for the 3-class task
- Optional variation: freeze or fine-tune the backbone depending on compute budget

## 3. Experiments and Results

### 3.1 Experimental Setup

Common settings used for all models:

- Dataset: VOC2007 `{person, car, dog}`
- Image resize: `448 x 448`
- Train/test split: official VOC2007 `trainval` / `test`
- Metric: **mAP@0.5**

### 3.2 Quantitative Results

Replace the table values after training:

| Model / Strategy | Trainable Params (approx.) | Train Images | mAP@0.5 |
| --- | ---: | ---: | ---: |
| Frozen DINOv2 + custom head | `[...]` | `[...]` | `[...]` |
| Faster R-CNN ResNet-50 FPN | `[...]` | `[...]` | `[...]` |

### 3.3 Qualitative Results

Add 2-3 qualitative prediction examples for each model. Suggested structure:

- Figure 1: Frozen DINOv2 + head predictions on test images
- Figure 2: Faster R-CNN predictions on the same or similar test images

## 4. Discussion

Under limited compute and limited labeled data, the **frozen DINO + small head** strategy has a strong practical argument because the backbone is already pretrained on large-scale data with self-supervised learning, freezing it keeps memory use lower and reduces training time, and only a small number of parameters need to be optimized.

However, the comparison model may still perform better in detection because its architecture is explicitly optimized for localization and proposals. A standard Faster R-CNN head has stronger inductive bias for detection than a minimal grid head built on frozen features.

If more data and more compute were available, the strategic choice could change. Fine-tuning at least part of the DINO backbone would likely improve task alignment, and a stronger custom detection head or multi-scale feature design could better exploit the DINO representation.

These observations connect back to the broader course themes:

- **Backbones vs heads**: the backbone provides reusable visual representation, while the head adapts it to a specific task.
- **Supervised vs self-supervised pretraining**: DINO offers broad transferable features without task labels, whereas Faster R-CNN benefits from supervised detection priors.
- **Reuse of a single backbone across tasks**: the DINO setup highlights how one pretrained representation can support downstream tasks with lightweight task-specific heads.
