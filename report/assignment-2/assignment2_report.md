# Assignment 2: Object Detection with a Pretrained Backbone

## 1. Task and Data

This project studies object detection on the **PASCAL VOC 2007 detection dataset** using the restricted class subset **{person, car, dog}**. The implementation uses the official **trainval** split for training and the official **test** split for evaluation, with both splits filtered so that only images containing at least one of the three target classes are retained.

The data was extracted locally into two roots:

- Training root: `data/train-validation-data`
- Test root: `data/test-data`

The project audit trail confirms that both compared models used the same filtered splits:

- Classes: `person`, `car`, `dog`
- Training images used: **1000**
- Test images used after filtering: **2895**
- Subsampling applied: **yes**, a reproducible random subset of 1000 trainval images with seed `42`

The exact subset identity is persisted in the audit manifests:

- Train subset SHA-256: `28a4ca86db9f501eb0cee8f9403461776aab6e8d6200391b707564073273017e`
- Test subset SHA-256: `71b8e73e8ebdf89bc5044cae094c6d6f69fa7564706ac39f3f76feafea5ce6f2`

This audit trail is useful because it verifies that both models were configured against the same train subset and exactly the same test set.

## 2. Model Designs

### 2.1 Main Model: Frozen DINOv2 + Custom Detection Head

The main model uses a pretrained **DINOv2 small** backbone (`facebook/dinov2-small`) as a **frozen feature extractor**. The encoder is loaded through the Hugging Face `transformers` implementation and all backbone parameters are frozen, so optimization only updates the task-specific head.

Feature extraction works as follows:

- The input image is resized to `448 x 448`.
- The DINOv2 backbone produces a sequence of patch tokens.
- The class token is discarded.
- The remaining patch tokens are reshaped into a 2D spatial feature map.

On top of that feature map, the project implements a custom **grid-based detection head**:

- A small convolutional stem adapts the frozen DINO features.
- A classification branch predicts one of `{background, person, car, dog}` per spatial location.
- A regression branch predicts normalized bounding boxes in `(cx, cy, w, h)` form.
- During training, each object is assigned to the grid cell containing its box center.

Training strategy:

- Backbone parameters frozen: **all DINOv2 backbone weights**
- Trainable parameters: **custom detection head only**
- Approximate trainable parameters: **1,477,128**
- Total parameters: **23,533,704**

Losses:

- Cross-entropy classification loss
- L1 box regression loss
- Generalized IoU loss

This design intentionally reflects a compute-constrained setting: the expensive pretrained representation is reused as-is, while only a lightweight detector head is optimized.

### 2.2 Comparison Model: Faster R-CNN with ResNet-50 FPN

The comparison strategy is a **supervised ResNet-based detector** implemented with **Faster R-CNN + ResNet-50 FPN** from `torchvision`.

The model uses a pretrained supervised Faster R-CNN backbone and replaces the ROI classifier/predictor so it predicts the assignment classes. Although the detector internally uses a background category for no-object regions, the actual object categories being learned and evaluated remain `person`, `car`, and `dog`.

Training strategy:

- Backbone/model: **Faster R-CNN with ResNet-50 FPN**
- Fine-tuning: the detector is configured to fine-tune the backbone and detection head together
- Approximate trainable parameters: **41,087,011**
- Total parameters: **41,309,411**

Compared with the frozen DINO approach, this baseline is much more detection-specialized but also much heavier to train.

## 3. Experiments and Results

### 3.1 Experimental Setup

Common settings used across the project:

- Dataset: VOC2007 filtered to `{person, car, dog}`
- Train split: official `trainval`
- Test split: official `test`
- Train subset size: `1000`
- Image resize: `448 x 448`
- Metric: **mAP@0.5**
- Environment used in recorded runs: **CPU-only**

The project also logs an audit trail for reproducibility:

- dataset manifests with exact image ids and digests
- run manifests with model configuration and parameter counts
- checkpoint and training-progress metadata

### 3.2 Quantitative Results

The latest saved project artifacts show a completed partial training run for the DINO model and a configured Faster R-CNN baseline run that has not yet produced saved evaluation metrics.

| Model / Strategy | Trainable Params (approx.) | Train Images | Test Images | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv2-small + custom grid head | 1,477,128 | 1000 | 2895 | 0.000 |
| Faster R-CNN ResNet-50 FPN | 41,087,011 | 1000 | 2895 | Not yet recorded |

Additional notes from the saved artifacts:

- DINO run environment: CPU-only
- DINO training progress recorded through epoch **4**
- Best recorded DINO checkpoint: epoch **1**
- Faster R-CNN run manifest exists, which confirms the model configuration and dataset alignment, but no saved checkpoint or evaluation report is available yet

### 3.3 Qualitative Results

At the time of this report update, no saved qualitative prediction images were present in the project outputs directory for either model.

The visualization pipeline is implemented and ready to export predictions for 2-3 test images per model once checkpoints are available. The report should ultimately include:

- Figure 1: DINOv2 + custom head predictions on selected test images
- Figure 2: Faster R-CNN predictions on selected test images

Because the latest saved artifacts do not yet include exported `.png` prediction figures, these qualitative examples remain outstanding.

## 4. Discussion

### 4.1 Under Limited Compute and Limited Labeled Data

Under a limited-compute, limited-label regime, I would still prefer the **frozen DINO + small head** strategy as the more practical starting point, even though the current recorded result is weak.

The reasons are strategic rather than purely score-based:

- It trains far fewer parameters than Faster R-CNN.
- It is simpler to run within restricted hardware, especially on CPU or modest Colab settings.
- It makes stronger use of reusable pretrained representation learning.

That said, the current project evidence also shows the main risk of this strategy: a minimal custom detection head on top of frozen features may not be strong enough to deliver competitive localization performance without further refinement. In the saved run, the DINO model reached `mAP@0.5 = 0.0`, which suggests that simply freezing the backbone and attaching a small grid head is not automatically sufficient for this detection task.

By contrast, Faster R-CNN is more likely to perform better once training is completed because its architecture is explicitly designed for object detection, region proposal, and localization. So if the goal is the best detection performance rather than the most lightweight adaptation, the comparison strategy is still very attractive.

### 4.2 If More Data and More Compute Were Available

If more labeled data and more compute were available, I would shift away from the fully frozen setup and move toward a stronger fine-tuning strategy:

- unfreeze part of the DINO backbone, especially later blocks
- add a stronger detection head, ideally multi-scale rather than a very simple single-scale grid head
- compare that against a fully trained supervised detector baseline

With more compute, the Faster R-CNN baseline also becomes easier to justify, because its larger trainable parameter count is less of a bottleneck. In that regime, the best choice would likely be whichever detector yields stronger test-set localization after proper training rather than whichever is cheaper to optimize.

### 4.3 Connection Back to Assignment 1 Themes

This project connects clearly to the Assignment 1 themes.

**Backbones vs heads**

The backbone provides general-purpose visual features, while the head defines the task-specific mapping from representation to detection outputs. The DINO model makes this distinction very explicit: the backbone is reused unchanged, and almost all task adaptation is delegated to the head.

**Supervised vs self-supervised pretraining**

DINOv2 is a self-supervised pretrained backbone, so it offers strong transferable visual features without direct detection supervision. Faster R-CNN with a supervised ResNet backbone, on the other hand, benefits from a more detection-oriented supervised training lineage. The current project outcome suggests that good general representation alone is not enough; the downstream detection architecture still matters a great deal.

**Reusing a single backbone for many tasks**

The DINO setup is a direct example of backbone reuse: one pretrained visual encoder can be adapted to detection by swapping in a lightweight downstream head. This is conceptually powerful even when the first simple head is not yet strong enough, because it demonstrates how a single representation can support multiple tasks with relatively small task-specific additions.

## 5. Conclusion

The project successfully implements the required two-strategy comparison:

1. A **frozen DINOv2-small backbone with a custom grid-based detection head**
2. A **Faster R-CNN ResNet-50 FPN supervised baseline**

The latest saved audit artifacts confirm that both models were configured on the same 1000-image training subset and the same 2895-image filtered VOC2007 test set. The DINO run completed partial training on CPU and currently records `mAP@0.5 = 0.0`, while the Faster R-CNN baseline configuration is in place but does not yet have saved evaluation metrics or qualitative exports in the current workspace.

The main strategic lesson so far is that a frozen pretrained backbone is appealing under tight compute budgets, but the quality of the downstream detection head is critical. A strong reusable backbone helps, but detection performance still depends heavily on how effectively the head translates those features into localization and classification outputs.
