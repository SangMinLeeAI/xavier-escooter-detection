from torchvision import models
import torch.nn as nn
import torch


def get_scooter_model(
    device="cpu", weight_path="fine_tuned_mobilenetv3_with_scooter.pth"
):
    model = models.mobilenet_v3_large()
    model.classifier[-1] = nn.Linear(1280, 1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model
