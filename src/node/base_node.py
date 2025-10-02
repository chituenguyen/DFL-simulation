"""
Base DFL Node Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class BaseNode:
    """
    Base class for Decentralized Federated Learning nodes
    Enhanced with P2P communication and neighbor management
    """

    def __init__(self, node_id: int, model: nn.Module, dataloader: DataLoader,
                 config: Dict[str, Any]):
        """
        Initialize a DFL node

        Args:
            node_id: Unique identifier for the node
            model: Neural network model
            dataloader: Local training data
            config: Training configuration
        """
        self.node_id = node_id
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'rounds': []
        }

        # P2P Communication
        self.neighbors = []
        self.neighbor_models = {}
        self.communication_round = 0

        logger.info(f"Node {node_id} initialized with {len(dataloader.dataset)} samples")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_name = self.config.get('optimizer', 'sgd').lower()
        lr = self.config.get('learning_rate', 0.01)
        momentum = self.config.get('momentum', 0.9)
        weight_decay = self.config.get('weight_decay', 0.0005)

        if optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def local_train(self, epochs: int) -> Dict[str, float]:
        """
        Perform local training for specified epochs

        Args:
            epochs: Number of local training epochs

        Returns:
            Training metrics (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

        avg_loss = total_loss / (epochs * len(self.dataloader))
        accuracy = correct / total

        # Update history
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)

        logger.info(f"Node {self.node_id} - Local training complete: "
                   f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total,
            'node_id': self.node_id
        }

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters with proper device handling"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name].to(self.device))

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data

        Args:
            test_loader: Test data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }

    def get_data_size(self) -> int:
        """Get the size of local dataset"""
        return len(self.dataloader.dataset)

    def set_neighbors(self, neighbors: list) -> None:
        """Set neighbor node IDs for P2P communication"""
        self.neighbors = neighbors
        logger.info(f"Node {self.node_id} neighbors set to: {neighbors}")

    def get_neighbors(self) -> list:
        """Get list of neighbor node IDs"""
        return self.neighbors

    def prepare_model_for_sharing(self) -> Dict[str, torch.Tensor]:
        """Prepare model parameters for sharing with neighbors"""
        return {
            'parameters': self.get_model_parameters(),
            'data_size': self.get_data_size(),
            'node_id': self.node_id,
            'round': self.communication_round
        }

    def update_from_neighbors(self, neighbor_updates: Dict[int, Dict]) -> None:
        """Update model based on neighbor models using weighted averaging"""
        if not neighbor_updates:
            logger.warning(f"Node {self.node_id}: No neighbor updates received")
            return

        # Collect all model parameters and weights
        all_params = [self.get_model_parameters()]
        weights = [self.get_data_size()]

        for neighbor_id, update in neighbor_updates.items():
            if 'parameters' in update:
                all_params.append(update['parameters'])
                weights.append(update.get('data_size', self.get_data_size()))

        # Weighted averaging
        total_weight = sum(weights)
        averaged_params = {}

        for param_name in all_params[0].keys():
            weighted_sum = sum(
                params[param_name] * weight
                for params, weight in zip(all_params, weights)
            )
            averaged_params[param_name] = weighted_sum / total_weight

        # Update model
        self.set_model_parameters(averaged_params)
        self.communication_round += 1

        logger.info(f"Node {self.node_id} aggregated with {len(neighbor_updates)} neighbors")