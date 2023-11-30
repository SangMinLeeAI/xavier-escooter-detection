import time
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.models import MobileNetV3
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torchvision import transforms

from src.utils import plot_image


def resize_box(box: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Resize the box coordinates."""
    box[0] -= int((box[2] - box[0]) * factor)
    box[1] -= int((box[3] - box[1]) * factor)
    box[2] += int((box[2] - box[0]) * factor)
    box[3] += int((box[3] - box[1]) * factor)
    return box


def crop_and_transform(image: Image.Image, box: np.ndarray, device: str) -> Tensor:
    """Crop the image using the given box coordinates and transform it to a tensor."""
    box_image = image.crop(box)
    box_image_tensor = transforms.ToTensor()(box_image).to(device)
    return box_image_tensor


def change_image_for_scooter(image_tensor: Tensor) -> Tensor:
    """Resize the image tensor for scooter model."""
    return F.resize(image_tensor, (224, 224)).unsqueeze(0)


def is_person_valid(label: int, score: float) -> bool:
    """Check if the label corresponds to a valid person."""
    return label == 1 and score >= 0.5


def is_helmet_present(model: FasterRCNN, image_tensor: Tensor) -> bool:
    """Check if a helmet is present in the image."""
    with torch.no_grad():
        result = model(image_tensor.unsqueeze(0))
        helmet_labels = result[0]["labels"].tolist()
        return 1 in helmet_labels or 2 in helmet_labels or 3 in helmet_labels


def is_scooter_present(model: MobileNetV3, image_tensor: Tensor) -> bool:
    """Check if a scooter is present in the image."""
    with torch.no_grad():
        result = model(change_image_for_scooter(image_tensor))
        predicted = (torch.sigmoid(result) > 0.2).float().item()
        return predicted == 1


def inference(
    helmet_model: FasterRCNN,
    person_model: FasterRCNN,
    scooter_model: MobileNetV3,
    image: Image.Image,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Perform inference on the given image."""
    helmet_model.eval().to(device)
    person_model.eval().to(device)
    scooter_model.eval().to(device)

    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        person_result = person_model(image_tensor)

    boxes: np.ndarray = person_result[0]["boxes"].data.cpu().numpy()
    labels: np.ndarray = person_result[0]["labels"].data.cpu().numpy()
    scores: np.ndarray = person_result[0]["scores"].data.cpu().numpy()

    valid_indices: List[int] = [
        index
        for index, (label, score) in enumerate(zip(labels, scores))
        if is_person_valid(label, score)
    ]

    for index in valid_indices:
        box = resize_box(boxes[index])
        box_image_tensor = crop_and_transform(image, box, device)

        if is_helmet_present(helmet_model, box_image_tensor):
            labels[index] = 2
        elif is_scooter_present(scooter_model, box_image_tensor):
            labels[index] = 3

    return (
        [boxes[i] for i in valid_indices],
        [labels[i] for i in valid_indices],
        [scores[i] for i in valid_indices],
    )


def get_live_inference_from_camera(
    helmet_model: FasterRCNN,
    person_model: FasterRCNN,
    scooter_model: MobileNetV3,
    class_labels: List[str],
    device: str = "cpu",
    camera_id: int = 0,
):
    cap = cv2.VideoCapture(camera_id)
    cap.set(3, 640)
    cap.set(4, 480)
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(frame)
            image_tensor = (
                transforms.ToTensor()(image).to("cpu").permute(1, 2, 0).numpy()
            )

            # Assuming the inference function returns bounding boxes in the format [x_min, y_min, x_max, y_max]
            boxes, labels, scores = inference(
                helmet_model, person_model, scooter_model, image
            )

            for box, label, score in zip(boxes, labels, scores):
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Class: {class_labels[label]}, Score: {score:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Display the result
            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
