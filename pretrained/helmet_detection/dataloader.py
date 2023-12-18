import os
import re
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .utils import TEST_SIZE, SEED


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(os.path.join(root_dir, "Images"))
            if f.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, "Images", img_name)
        annotation_path = os.path.join(
            self.root_dir, "Annotations", os.path.splitext(img_name)[0] + ".xml"
        )

        image = Image.open(img_path).convert("RGB")
        target = self.parse_annotation(annotation_path)

        image = F.to_tensor(image).to("cuda")
        if self.transform:
            image = self.transform.random_adjust_contrast(image, enable=True)
            image = self.transform.random_adjust_brightness(image, enable=True)
        return image, target

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Skip boxes with zero width or height
            if xmax <= xmin or ymax <= ymin or xmax == 0 or ymax == 0:
                continue

            class_mapping = {
                "helmet": 1,
                "head_with_helmet": 2,
                "person_with_helmet": 3,
                "head": 4,
                "person_no_helmet": 5,
                "face": 6,
                "person": 7,
            }

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_mapping[label])

        extracted_int = int(re.search(r"\d+", root.find("filename").text).group())
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(
                [extracted_int]
            ),
            "area": torch.tensor(
                [(xmax - xmin) * (ymax - ymin) for xmin, ymin, xmax, ymax in boxes],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return target


"""
Class that holds all the augmentation related attributes
"""


class Transformation:
    def get_probability(self):
        return np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    def random_adjust_contrast(self, image, enable=None):
        enable = self.get_probability() if enable is None else enable
        return F.adjust_contrast(image, 2) if enable else image

    def random_adjust_brightness(self, image, enable=None):
        enable = enable = self.get_probability() if enable is None else enable
        return F.adjust_brightness(image, 2) if enable else image

    def random_hflip(self, image, boxes, enable=None):
        enable = enable = self.get_probability() if enable is None else enable
        if enable:
            # flip image
            new_image = F.hflip(image)

            # flip boxes
            new_boxes = boxes.clone()
            new_boxes[:, 0] = image.shape[2] - boxes[:, 0]  # image width - xmin
            new_boxes[:, 2] = image.shape[2] - boxes[:, 2]  # image_width - xmax
            new_boxes = new_boxes[
                :, [2, 1, 0, 3]
            ]  
            return new_image, new_boxes
        else:
            return image, boxes


def collate_fn(batch):
    return tuple(zip(*batch))


def transform(image, target):
    image = F.to_tensor(image)
    return image, target


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def get_data_loader(
    root_dir, batch_size=4, num_workers=4, pin_memory=True, is_test=False
):
    torch.manual_seed(SEED)
    dataset = CustomObjectDetectionDataset(root_dir, transform=Transformation())
    indices = torch.randperm(len(dataset)).tolist()

    test_size = int(len(dataset) * TEST_SIZE)
    dataset = torch.utils.data.Subset(dataset, indices[:-test_size])

    if is_test:
        dataset = CustomObjectDetectionDataset(root_dir, transform=None)
        dataset_test = torch.utils.data.Subset(dataset, indices[-test_size:])
        data_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        return data_loader

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return data_loader
