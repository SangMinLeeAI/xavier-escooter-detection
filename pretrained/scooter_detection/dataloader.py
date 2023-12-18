# dataloader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BinaryImageDataset

def get_train_loader() -> DataLoader:
    # Specify the path to your dataset
    training_dir = "./xavier-escooter/e-scooter rider dataset/training"

    # Create an instance of the BinaryImageDataset
    binary_dataset = BinaryImageDataset(root_dir=training_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))

    # Create a DataLoader
    train_loader = DataLoader(binary_dataset, batch_size=8, shuffle=True, num_workers=0)
    return train_loader

def get_val_loader() -> DataLoader:
    # Specify the path to your dataset
    testing_dir = "./xavier-escooter/e-scooter rider dataset/testing"

    # Create an instance of the BinaryImageDataset
    testing_binary_dataset = BinaryImageDataset(root_dir=testing_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))

    # Create a DataLoader
    val_loader = DataLoader(testing_binary_dataset, batch_size=8, shuffle=True, num_workers=0)
    return val_loader
