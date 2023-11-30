from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.models import MobileNetV3
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torchvision import transforms


def change_image_for_scooter(image: Tensor) -> Tensor:
    image: Tensor = F.resize(image, (224, 224))
    image: Tensor = image.unsqueeze(0)
    return image


def inference(
    helmet_model: FasterRCNN,
    person_model: FasterRCNN,
    scooter_model: MobileNetV3,
    image: Image,
    device: str = "cpu",
):
    # 초기 세팅
    helmet_model.eval()
    person_model.eval()
    scooter_model.eval()

    helmet_model.to(device)
    person_model.to(device)
    scooter_model.to(device)

    # 먼저 사람인지 인식하고 사람이면 스쿠터를 탔는지 확인 후 헬멧을 썼는지 확인

    # 이미지 불러오기

    image_tensor: Tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # 사람인지 확인
    person_result: List[Dict[str, Tensor]] = person_model(image_tensor)
    boxes: np.ndarray = person_result[0]["boxes"].data.cpu().numpy()
    labels: np.ndarray = person_result[0]["labels"].data.cpu().numpy()
    scores: np.ndarray = person_result[0]["scores"].data.cpu().numpy()
    to_delete: List = []
    person_with_helmet: Dict[int, bool] = {}
    with torch.no_grad():
        for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if label > 1 or score < 0.5:  # 사람이 아니면
                to_delete.append(index)
            else:
                ## box 좌표를 이용해서 box 범위를 30% 넓힌다
                box[0] = box[0] - int((box[2] - box[0]) * 0.7)
                box[1] = box[1] - int((box[3] - box[1]) * 0.3)
                box[2] = box[2] + int((box[2] - box[0]) * 0.7)
                box[3] = box[3] + int((box[3] - box[1]) * 0.3)
                # 박스를 이미지로 자른다
                box_image: Image = image.crop(box)
                box_image_tensor: Tensor = transforms.ToTensor()(box_image).to(device)
                # 스쿠터 탔는지 확인
                helmet_result: List[Dict[str, Tensor]] = helmet_model(
                    box_image_tensor.unsqueeze(0)
                )
                helmet_labels: List = helmet_result[0]["labels"].tolist()
                if 1 in helmet_labels:
                    labels[index] = 2
                else:
                    scooter_result: Tensor = scooter_model(
                        change_image_for_scooter(box_image_tensor)
                    )
                    predicted: float = (torch.sigmoid(scooter_result) > 0.2).float().item()
                    if predicted == 1:
                        labels[index] = 3

    box_result: List[int] = [i for j, i in enumerate(boxes) if j not in to_delete]
    label_result: List[int] = [i for j, i in enumerate(labels) if j not in to_delete]
    score_result: List[int] = [i for j, i in enumerate(scores) if j not in to_delete]
    return box_result, label_result, score_result
