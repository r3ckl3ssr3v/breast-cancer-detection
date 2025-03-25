import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms

class MammogramDataset(Dataset):
    """Mammogram dataset for PyTorch."""
    
    def __init__(self, csv_file, transform=None, num_classes=3):
        """
        Args:
            csv_file: Path to the CSV file with image paths and labels
            transform: Optional transform to be applied on a sample
            num_classes: Number of classes (3 for normal/benign/malignant)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_frame.iloc[idx, 0]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to RGB (3 channels) by duplicating the grayscale channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        # One-hot encode the label if needed
        if self.num_classes > 2:
            label_onehot = torch.zeros(self.num_classes)
            label_onehot[label] = 1
            return image, label_onehot
        
        return image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing processed data and CSV files
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MammogramDataset(
        csv_file=os.path.join(data_dir, 'train.csv'),
        transform=train_transform
    )
    
    val_dataset = MammogramDataset(
        csv_file=os.path.join(data_dir, 'val.csv'),
        transform=val_test_transform
    )
    
    test_dataset = MammogramDataset(
        csv_file=os.path.join(data_dir, 'test.csv'),
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader