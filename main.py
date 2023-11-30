import torch

import pretrained
from src.inference import get_live_inference_from_camera

device = "cuda" if torch.cuda.is_available() else "cpu"

helmet_model = pretrained.helmet_detection.model.get_helmet_model(
    weight_path="pretrained/helmet_detection/fine_tuned_fastercnn_helmet_detection.pth"
)
person_model = pretrained.person_detection.model.get_person_model(
    weight_path="pretrained/person_detection/fine_tuned_fastercnn_person_detection.pth"
)
scooter_model = pretrained.scooter_detection.model.get_scooter_model(
    weight_path="pretrained/scooter_detection/fine_tuned_mobilenetv3_with_scooter.pth"
)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    get_live_inference_from_camera(
        helmet_model=helmet_model,
        person_model=person_model,
        scooter_model=scooter_model,
        class_labels=["Background", "person", "person_with_helmet", "criminal"],
        device=device,
    )
