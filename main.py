import pretrained
from src.inference import inference
from src.utils import plot_image
import torch
from PIL import Image
from torchvision import transforms
import requests

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
    image = Image.open("example_with_helmet.jpeg").convert("RGB")
    image_tensor = transforms.ToTensor()(image).to("cpu").permute(1, 2, 0).numpy()
    boxes, labels, scores = inference(helmet_model, person_model, scooter_model, image)
    class_labels = ["Background", "person", "person_with_helmet", "criminal"]
    print(labels)
    plot_image(
        img = image_tensor,
        boxes = boxes,
        scores= scores,
        labels= labels,
        class_label= class_labels)
