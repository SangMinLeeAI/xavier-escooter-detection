from typing import List, Dict

import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torchvision.models import MobileNetV3
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F

import pretrained


def resize_box(box: ndarray, factor: float = 0.3) -> ndarray:
    """Resize the box coordinates."""
    box[0] -= int((box[2] - box[0]) * factor)
    box[1] -= int((box[3] - box[1]) * factor)
    box[2] += int((box[2] - box[0]) * factor)
    box[3] += int((box[3] - box[1]) * factor)
    return box


def crop_and_transform(image: Image, box: ndarray, device: str) -> Tensor:
    """Crop the image using the given box coordinates and transform it to a tensor."""
    box_image = image.crop(box)
    box_image_tensor = F.toTensor()(box_image).to(device)
    return box_image_tensor


def is_person_detected(result: List[Dict[Tensor]]) -> bool:
    """Check if a person is detected."""
    boxes = result[0]["boxes"].cpu().numpy().astype(int)
    labels = result[0]["labels"].cpu().numpy()
    scores = result[0]["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5 and label == 1:
            return True
    return False


def is_scooter_detected(model: MobileNetV3, image_tensor: Tensor) -> bool:
    """Check if a scooter is detected."""
    with torch.no_grad():
        result = model(image_tensor)
        predicted = (torch.sigmoid(result) > 0.5).float()
        return predicted == 1


def is_helmet_detected(model: FasterRCNN, image_tensor: Tensor) -> bool:
    """Check if a helmet is detected."""
    with torch.no_grad():
        result = model(image_tensor)
        helmet_labels = result[0]["labels"].cpu().numpy()
        return 1 in helmet_labels


def inference(
    helmet_model: FasterRCNN,
    person_model: FasterRCNN,
    scooter_model: MobileNetV3,
    image_path: str,
    device: str = "cpu",
) -> int:
    """Perform inference on the given image."""
    helmet_model.eval()
    person_model.eval()
    scooter_model.eval()

    image = Image.open(image_path)
    image_tensor = F.toTensor()(image).to(device)

    with torch.no_grad():
        person_inference_result = person_model(image_tensor)
    if is_person_detected(person_inference_result):
        boxes = person_inference_result[0]["boxes"].cpu().numpy().astype(int)

        for box in boxes:
            resized_box = resize_box(box)
            box_image_tensor = crop_and_transform(image, resized_box, device)

            if is_scooter_detected(scooter_model, box_image_tensor):
                if is_helmet_detected(helmet_model, box_image_tensor):
                    return 1

    return 0
