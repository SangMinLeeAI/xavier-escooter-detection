import time
from typing import Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CustomModel
from dataloader import get_train_loader, get_val_loader
from utils import save_model_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += predicted.eq(labels.float()).sum().item()

            pbar.update(1)
            pbar.set_postfix({'Loss': loss.item(), 'Acc': correct / total * 100})

    epoch_loss = train_loss / total
    epoch_acc = correct / total * 100
    print(f"Train | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% ({correct}/{total})")
    return epoch_loss, epoch_acc

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Validation') as pbar:
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += predicted.eq(labels.float()).sum().item()

                pbar.update(1)

        epoch_loss = val_loss / total
        epoch_acc = correct / total * 100
        print(f"Validation | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% ({correct}/{total})")

    return epoch_loss, epoch_acc

def main():
    # Initialize your model, criterion, optimizer, and dataloaders
    model = CustomModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    # Training loop
    epoch_length = 5
    best_acc = 0

    for epoch in range(epoch_length):
        print(f"Epoch {epoch}")
        epoch_time = time.time()

        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        learning_time = time.time() - epoch_time
        print(f'**Epoch time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s')

        # Save model
        if val_acc > best_acc:
            best_acc = val_acc
            save_model_checkpoint(model, optimizer, epoch, val_acc, 'model_checkpoint.pth')

    print('Training completed!')

if __name__ == "__main__":
    main()
