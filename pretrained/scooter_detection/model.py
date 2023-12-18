import torch
from torch import nn
from torchvision.models import efficientnet_v2_m

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        weights = efficientnet_v2_m.EfficientNetV2MWeights.DEFAULT
        base_model = efficientnet_v2_m(weights=weights)
        base_model.classifier[-1] = nn.Linear(1280, 1)
        self.model = nn.DataParallel(base_model, device_ids=[0, 1, 2, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
