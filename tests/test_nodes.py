"""
Tests for node implementations
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from node import BaseNode, NodeManager
from models import ResNet18


class TestBaseNode(unittest.TestCase):
    """Test BaseNode implementation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create simple model
        self.model = ResNet18(num_classes=10)

        # Create dummy dataset
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Configuration
        self.config = {
            'device': 'cpu',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'optimizer': 'sgd'
        }

        self.node_id = 0

    def test_node_initialization(self):
        """Test node initialization"""
        node = BaseNode(
            node_id=self.node_id,
            model=self.model,
            dataloader=self.dataloader,
            config=self.config
        )

        self.assertEqual(node.node_id, self.node_id)
        self.assertEqual(node.model, self.model)
        self.assertEqual(node.dataloader, self.dataloader)
        self.assertEqual(node.config, self.config)
        self.assertIsNotNone(node.optimizer)

    def test_optimizer_creation(self):
        """Test optimizer creation"""
        # Test SGD optimizer
        config_sgd = {**self.config, 'optimizer': 'sgd'}
        node = BaseNode(self.node_id, self.model, self.dataloader, config_sgd)
        self.assertIsInstance(node.optimizer, torch.optim.SGD)

        # Test Adam optimizer
        config_adam = {**self.config, 'optimizer': 'adam'}
        node = BaseNode(self.node_id, self.model, self.dataloader, config_adam)
        self.assertIsInstance(node.optimizer, torch.optim.Adam)

        # Test unsupported optimizer
        config_invalid = {**self.config, 'optimizer': 'invalid'}
        with self.assertRaises(ValueError):
            BaseNode(self.node_id, self.model, self.dataloader, config_invalid)

    def test_local_training(self):
        """Test local training"""
        node = BaseNode(
            node_id=self.node_id,
            model=self.model,
            dataloader=self.dataloader,
            config=self.config
        )

        # Train for 1 epoch
        metrics = node.local_train(epochs=1)

        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('samples', metrics)

        # Check that loss and accuracy are reasonable
        self.assertGreaterEqual(metrics['loss'], 0)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertEqual(metrics['samples'], len(self.dataloader.dataset))

        # Check that training history is updated
        self.assertEqual(len(node.history['train_loss']), 1)
        self.assertEqual(len(node.history['train_accuracy']), 1)

    def test_model_parameter_operations(self):
        """Test model parameter get/set operations"""
        node = BaseNode(
            node_id=self.node_id,
            model=self.model,
            dataloader=self.dataloader,
            config=self.config
        )

        # Get initial parameters
        initial_params = node.get_model_parameters()
        self.assertIsInstance(initial_params, dict)
        self.assertGreater(len(initial_params), 0)

        # Create modified parameters
        modified_params = {}
        for name, param in initial_params.items():
            modified_params[name] = param + 0.1

        # Set modified parameters
        node.set_model_parameters(modified_params)

        # Verify parameters were updated
        new_params = node.get_model_parameters()
        for name in initial_params.keys():
            self.assertFalse(torch.equal(initial_params[name], new_params[name]))

    def test_evaluation(self):
        """Test model evaluation"""
        node = BaseNode(
            node_id=self.node_id,
            model=self.model,
            dataloader=self.dataloader,
            config=self.config
        )

        # Evaluate model
        metrics = node.evaluate(self.dataloader)

        # Check metrics
        self.assertIn('test_loss', metrics)
        self.assertIn('test_accuracy', metrics)

        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['test_loss'], 0)
        self.assertGreaterEqual(metrics['test_accuracy'], 0)
        self.assertLessEqual(metrics['test_accuracy'], 1)

    def test_data_size(self):
        """Test getting data size"""
        node = BaseNode(
            node_id=self.node_id,
            model=self.model,
            dataloader=self.dataloader,
            config=self.config
        )

        size = node.get_data_size()
        self.assertEqual(size, len(self.dataloader.dataset))


class TestNodeManager(unittest.TestCase):
    """Test NodeManager implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'data': {
                'data_dir': './test_data',
                'batch_size': 32,
                'num_workers': 2
            },
            'partition': {
                'num_nodes': 3,
                'alpha': 0.5
            },
            'model': {
                'num_classes': 10,
                'pretrained': False
            },
            'training': {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'optimizer': 'sgd'
            },
            'experiment': {
                'seed': 42
            }
        }

    def test_node_manager_initialization(self):
        """Test node manager initialization"""
        manager = NodeManager(self.config)
        self.assertEqual(manager.config, self.config)
        self.assertEqual(len(manager.nodes), 0)
        self.assertIsNone(manager.test_loader)

    @patch('src.data.CIFAR10Loader.load_data')
    @patch('src.data.CIFAR10Loader.create_dataloader')
    def test_create_nodes_mocked(self, mock_create_dataloader, mock_load_data):
        """Test node creation with mocked data loading"""
        # Mock datasets
        mock_train_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_train_dataset.targets = list(range(10)) * 100

        mock_test_dataset = Mock()
        mock_load_data.return_value = (mock_train_dataset, mock_test_dataset)

        # Mock dataloaders
        mock_dataloader = Mock()
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.__len__ = Mock(return_value=100)
        mock_create_dataloader.return_value = mock_dataloader

        manager = NodeManager(self.config)
        nodes = manager.create_nodes()

        # Check that nodes were created
        self.assertEqual(len(nodes), self.config['partition']['num_nodes'])
        self.assertEqual(len(manager.nodes), self.config['partition']['num_nodes'])

        # Check that all nodes are BaseNode instances
        for node in nodes:
            self.assertIsInstance(node, BaseNode)

    def test_get_node_data_sizes(self):
        """Test getting node data sizes"""
        manager = NodeManager(self.config)

        # Mock nodes
        mock_nodes = []
        for i in range(3):
            mock_node = Mock()
            mock_node.node_id = i
            mock_node.get_data_size.return_value = (i + 1) * 100
            mock_nodes.append(mock_node)

        manager.nodes = mock_nodes

        data_sizes = manager.get_node_data_sizes()
        expected = {0: 100, 1: 200, 2: 300}
        self.assertEqual(data_sizes, expected)


if __name__ == '__main__':
    unittest.main()