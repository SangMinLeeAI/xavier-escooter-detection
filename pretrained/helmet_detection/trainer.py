import torch

from utils import LR, LR_MOMENTUM, LR_DECAY_RATE, LR_SCHED_STEP_SIZE, LR_SCHED_GAMMA
import os
import re
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import utils

from datetime import datetime
from tqdm import tqdm

import pickle

from torchvision import transforms as T

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.datasets import VisionDataset
from xml.etree import ElementTree as ET
from PIL import Image


def inference(img, model, detection_threshold=0.70):
    """
    Infernece of a single input image

    inputs:
      img: input-image as torch.tensor (shape: [C, H, W])
      model: model for infernce (torch.nn.Module)
      detection_threshold: Confidence-threshold for NMS (default=0.7)

    returns:
      boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
      labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
      scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
    """
    model.eval()
    img = img.to("cuda")
    outputs = model([img])

    boxes = outputs[0]["boxes"].data.cpu().numpy()
    scores = outputs[0]["scores"].data.cpu().numpy()
    labels = outputs[0]["labels"].data.cpu().numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    labels = labels[scores >= detection_threshold]
    scores = scores[scores >= detection_threshold]

    return boxes, scores, labels


def plot_image(img, boxes, scores, labels, dataset, save_path=None):
    """
    Function that draws the BBoxes, scores, and labels on the image.

    inputs:
      img: input-image as numpy.array (shape: [H, W, C])
      boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
      scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
      labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
      dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
    """

    cmap = plt.get_cmap("tab20b")
    class_labels = np.array(dataset)
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = img.shape
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img)
    for i, box in enumerate(boxes):
        class_pred = labels[i]
        conf = scores[i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            box[0],
            box[1],
            s=class_labels[int(class_pred)] + " " + str(int(100 * conf)) + "%",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Used to save inference phase results
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


"""
Function to train the model over one epoch.
"""


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    train_loss_list = []

    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    for idx, data in enumerate(tqdm_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]  # targets = {'boxes'=tensor, 'labels'=tensor}

        losses = model(images, targets)

        loss = sum(loss for loss in losses.values())

        loss_val = loss.item()
        train_loss_list.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        loss_int = loss.int()
        tqdm_bar.set_description(desc=f"Training Loss: {loss.item():.3f}")

    return train_loss_list


"""
Function to validate the model
"""


def evaluate(model, data_loader_test, device):
    val_loss_list = []

    tqdm_bar = tqdm(data_loader_test, total=len(data_loader_test))

    for i, data in enumerate(tqdm_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            losses = model(images, targets)

        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()
        val_loss_list.append(loss_val)

        tqdm_bar.set_description(desc=f"Validation Loss: {loss.item():.4f}")
    return val_loss_list


"""
Function to plot training and valdiation losses and save them in `output_dir'
"""


def plot_loss(train_loss, valid_loss, OUTPUT_DIR="."):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    train_ax.plot(train_loss, color="blue")
    train_ax.set_xlabel("Iteration")
    train_ax.set_ylabel("Training Loss")

    valid_ax.plot(valid_loss, color="red")
    valid_ax.set_xlabel("Iteration")
    valid_ax.set_ylabel("Validation loss")

    figure_1.savefig(f"{OUTPUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUTPUT_DIR}/valid_loss.png")


def get_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=LR_MOMENTUM, weight_decay=LR_DECAY_RATE
    )
    return optimizer


def get_lr_scheduler(optimizer):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_SCHED_STEP_SIZE, gamma=LR_SCHED_GAMMA
    )
    return lr_scheduler


def train(
    model=None,
    optimizer=None,
    lr_scheduler=None,
    num_epochs=None,
    data_loader=None,
    device=None,
    data_loader_test=None,
):
    current_dir = os.getcwd()

    # Fetch current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    output_dir_name = "output-" + dt_string

    OUTPUT_DIR = os.path.join(current_dir, output_dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find latest saved chcekpoint
    checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth")]
    if checkpoint_files:
        # Last ckpt file
        checkpoint_files.sort()
        latest_checkpoint = os.path.join(OUTPUT_DIR, checkpoint_files[-1])

        # Load the ckpt
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss_dict = checkpoint["loss_dict"]
    else:
        start_epoch = 0
        loss_dict = {"train_loss": [], "valid_loss": []}
    """
    Train the model over all epochs
    """
    for epoch in range(start_epoch, num_epochs):
        print("----------Epoch {}----------".format(epoch + 1))

        # Train the model for one epoch
        train_loss_list = train_one_epoch(model, optimizer, data_loader, device, epoch)
        loss_dict["train_loss"].extend(train_loss_list)

        lr_scheduler.step()

        # Run evaluation
        valid_loss_list = evaluate(model, data_loader_test, device)
        loss_dict["valid_loss"].extend(valid_loss_list)

        # Svae the model ckpt after every epoch
        ckpt_file_name = f"{OUTPUT_DIR}/epoch_{epoch + 1}_model.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_dict": loss_dict,
            },
            ckpt_file_name,
        )

        # NOTE: The losses are accumulated over all iterations
        plot_loss(loss_dict["train_loss"], loss_dict["valid_loss"])

    # Store the losses after the training in a pickle
    with open(f"{OUTPUT_DIR}/loss_dict.pkl", "wb") as file:
        pickle.dump(loss_dict, file)
    torch.save(model.state_dict(), "fine_tuned_fastercnn_helmet_detection.pth")
    print("Training Finished !")
