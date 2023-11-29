import torch
import torchvision


def get_person_model(
    device="cpu", weight_path="fine_tuned_fastercnn_person_detection.pth"
):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model
