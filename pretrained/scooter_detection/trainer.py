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


# Training
def train(epoch, model, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = (
            inputs.to(device),
            labels.to(device).view(-1, 1),
        )  # Reshape labels to [batch_size, 1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(
            outputs, labels.float()
        )  # Convert labels to float for BCEWithLogitsLoss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        predicted = (
            torch.sigmoid(outputs) > 0.5
        ).float()  # Convert logits to probabilities
        total += labels.size(0)
        correct += predicted.eq(labels.float()).sum().item()

    epoch_loss = train_loss / total
    epoch_acc = correct / total * 100
    print(
        "Train | Loss: %.4f Acc: %.2f%% (%s/%s)"
        % (epoch_loss, epoch_acc, correct, total)
    )
    return epoch_loss, epoch_acc


def test(epoch, model, criterion, optimizer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (
                inputs.to(device),
                labels.to(device).view(-1, 1),
            )  # Reshape labels to [batch_size, 1]
            outputs = model(inputs)
            loss = criterion(
                outputs, labels.float()
            )  # Convert labels to float for BCEWithLogitsLoss

            test_loss += loss.item() * inputs.size(0)
            predicted = (
                torch.sigmoid(outputs) > 0.5
            ).float()  # Convert logits to probabilities
            total += labels.size(0)
            correct += predicted.eq(labels.float()).sum().item()

        epoch_loss = test_loss / total
        epoch_acc = correct / total * 100
        print(
            "Test | Loss: %.4f Acc: %.2f%% (%s/%s)"
            % (epoch_loss, epoch_acc, correct, total)
        )
    return epoch_loss, epoch_acc


def get_criteria():
    return nn.BCEWithLogitsLoss()


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def run_training(model, criterion, optimizer, scheduler, epoch_length=30):
    start_time = time.time()
    best_acc = 0
    save_loss = {"train": [], "test": []}
    save_acc = {"train": [], "test": []}
    for epoch in range(epoch_length):
        print("Epoch %s" % epoch)
        epoch_time = time.time()
        train_loss, train_acc = train(epoch, model, criterion, optimizer)
        save_loss["train"].append(train_loss)
        save_acc["train"].append(train_acc)

        test_loss, test_acc = test(epoch, model, criterion, optimizer)
        save_loss["test"].append(test_loss)
        save_acc["test"].append(test_acc)

        scheduler.step()

        learning_time = time.time() - epoch_time
        print(f"**Epoch time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

        # Save model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)

    learning_time = time.time() - start_time
    print(f"**Learning time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    torch.save(model.state_dict(), "fine_tuned_mobilenetv3_full.pth")
