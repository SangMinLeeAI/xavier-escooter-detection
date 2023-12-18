from pprint import pprint

import numpy as np
import torch

import pretrained
from src.utils import plot_image
from src.inference import inference
from PIL import Image

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
    # get_live_inference_from_camera(
    #     helmet_model=helmet_model,
    #     person_model=person_model,
    #     scooter_model=scooter_model,
    #     class_labels=["Background", "person", "person_with_helmet", "criminal"],
    #     device=device,
    # )
    image_from_path = Image.open("./test.png").convert('RGB')
    image_numpy = np.array(image_from_path)
    results = inference(
        helmet_model=helmet_model,
        person_model=person_model,
        scooter_model=scooter_model,
        image=image_from_path,
        device=device,
    )
    pprint(results[1])

    plot_image(
        img = image_numpy,
        boxes = results[0],
        scores = results[2],
        labels = results[1],
        class_label = ["Background", "person", "person_with_helmet", "criminal"],
        save_path = None
    )
