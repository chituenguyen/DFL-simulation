"""
Tests for data loading and partitioning
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import CIFAR10Loader, DataPartitioner


class TestCIFAR10Loader(unittest.TestCase):
    """Test CIFAR-10 data loader"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_dir = "./test_data"
        self.batch_size = 32
        self.num_workers = 2

    def test_initialization(self):
        """Test loader initialization"""
        loader = CIFAR10Loader(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.assertEqual(loader.data_dir, self.data_dir)
        self.assertEqual(loader.batch_size, self.batch_size)
        self.assertEqual(loader.num_workers, self.num_workers)

    def test_transforms(self):
        """Test data transforms"""
        loader = CIFAR10Loader()

        # Test training transforms
        train_transform = loader.get_transforms(train=True)
        self.assertIsNotNone(train_transform)

        # Test test transforms
        test_transform = loader.get_transforms(train=False)
        self.assertIsNotNone(test_transform)

        # Transforms should be different
        self.assertNotEqual(str(train_transform), str(test_transform))

    @patch('torchvision.datasets.CIFAR10')
    def test_load_data(self, mock_cifar10):
        """Test data loading (mocked)"""
        # Mock dataset
        mock_dataset = Mock()
        mock_cifar10.return_value = mock_dataset

        loader = CIFAR10Loader()
        train_dataset, test_dataset = loader.load_data()

        # Check that CIFAR10 was called correctly
        self.assertEqual(mock_cifar10.call_count, 2)

    def test_dataloader_creation(self):
        """Test DataLoader creation"""
        loader = CIFAR10Loader(batch_size=16)

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)

        # Create dataloader
        dataloader = loader.create_dataloader(mock_dataset, shuffle=True)

        self.assertEqual(dataloader.batch_size, 16)
        self.assertTrue(dataloader.shuffle)


class TestDataPartitioner(unittest.TestCase):
    """Test data partitioner"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock dataset
        self.dataset = Mock()
        self.dataset.__len__ = Mock(return_value=1000)

        # Create mock labels (10 classes, 100 samples each)
        labels = []
        for class_id in range(10):
            labels.extend([class_id] * 100)
        self.dataset.targets = labels

        self.num_nodes = 5
        self.alpha = 0.5

    def test_initialization(self):
        """Test partitioner initialization"""
        partitioner = DataPartitioner(
            dataset=self.dataset,
            num_nodes=self.num_nodes,
            alpha=self.alpha
        )

        self.assertEqual(partitioner.dataset, self.dataset)
        self.assertEqual(partitioner.num_nodes, self.num_nodes)
        self.assertEqual(partitioner.alpha, self.alpha)

    def test_dirichlet_partition(self):
        """Test Dirichlet partitioning"""
        partitioner = DataPartitioner(
            dataset=self.dataset,
            num_nodes=self.num_nodes,
            alpha=self.alpha
        )

        # Partition data
        partitions = partitioner.partition_dirichlet()

        # Check number of partitions
        self.assertEqual(len(partitions), self.num_nodes)

        # Check that all samples are assigned
        total_samples = sum(len(partition.indices) for partition in partitions)
        self.assertEqual(total_samples, len(self.dataset))

        # Check no overlap between partitions
        all_indices = set()
        for partition in partitions:
            partition_indices = set(partition.indices)
            self.assertEqual(len(all_indices.intersection(partition_indices)), 0)
            all_indices.update(partition_indices)

    def test_iid_partition(self):
        """Test IID partitioning"""
        partitioner = DataPartitioner(
            dataset=self.dataset,
            num_nodes=self.num_nodes,
            alpha=self.alpha
        )

        # Partition data
        partitions = partitioner.partition_iid()

        # Check number of partitions
        self.assertEqual(len(partitions), self.num_nodes)

        # Check that all samples are assigned
        total_samples = sum(len(partition.indices) for partition in partitions)
        self.assertEqual(total_samples, len(self.dataset))

        # Check approximately equal partition sizes
        partition_sizes = [len(partition.indices) for partition in partitions]
        min_size = min(partition_sizes)
        max_size = max(partition_sizes)
        self.assertLessEqual(max_size - min_size, 2)  # Allow small variance

    def test_partition_with_labels_attribute(self):
        """Test partitioning with labels attribute instead of targets"""
        # Create dataset with labels attribute
        self.dataset.labels = self.dataset.targets
        delattr(self.dataset, 'targets')

        partitioner = DataPartitioner(
            dataset=self.dataset,
            num_nodes=self.num_nodes
        )

        partitions = partitioner.partition_dirichlet()
        self.assertEqual(len(partitions), self.num_nodes)

    def test_partition_without_labels(self):
        """Test partitioning fails without labels or targets"""
        # Remove labels
        delattr(self.dataset, 'targets')

        partitioner = DataPartitioner(
            dataset=self.dataset,
            num_nodes=self.num_nodes
        )

        with self.assertRaises(ValueError):
            partitioner.partition_dirichlet()


if __name__ == '__main__':
    unittest.main()