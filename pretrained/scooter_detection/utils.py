from typing import Tuple

import torch
from torch import nn, optim


def save_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, accuracy: float,
                          filename: str) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)


def load_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> Tuple[
    nn.Module, optim.Optimizer, int, float]:
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return model, optimizer, epoch, accuracy
