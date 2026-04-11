from __future__ import annotations

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..utils import VOC_CLASSES


def build_faster_rcnn(num_classes: int = len(VOC_CLASSES) + 1, train_backbone: bool = True):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if not train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    return model
