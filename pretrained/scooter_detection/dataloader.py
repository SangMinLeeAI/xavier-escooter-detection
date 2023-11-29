# pytorch
import torch
import torchvision
from torchvision import transforms, datasets, models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


##dataset
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# other
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


class BinaryImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir), reverse=True)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._make_dataset()

    def _make_dataset(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, class_idx


def transform_train():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
