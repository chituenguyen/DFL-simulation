"""
CIFAR-10 Data Loader with preprocessing
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List
import numpy as np


class CIFAR10Loader:
    """CIFAR-10 dataset loader with standard preprocessing"""

    def __init__(self, data_dir: str = "./data", batch_size: int = 128,
                 num_workers: int = 4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 normalization values
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]

    def get_transforms(self, train: bool = True) -> transforms.Compose:
        """Get data transforms for train/test"""
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

    def load_data(self) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
        """Load CIFAR-10 train and test datasets"""
        train_transform = self.get_transforms(train=True)
        test_transform = self.get_transforms(train=False)

        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        return train_dataset, test_dataset

    def create_dataloader(self, dataset: datasets.CIFAR10,
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader from dataset"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )