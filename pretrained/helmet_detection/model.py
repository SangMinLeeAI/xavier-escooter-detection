import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .utils import NUM_CLASSES


def get_helmet_model(
    device="cpu", weight_path="fine_tuned_fastercnn_helmet_detection.pth"
):
    num_classes = NUM_CLASSES
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model
