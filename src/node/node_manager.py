"""
Node Manager for DFL Lifecycle Management
"""

from typing import List, Dict, Any, Optional
import torch
import logging
from torch.utils.data import DataLoader

from .base_node import BaseNode
from ..models import create_model
from ..data import CIFAR10Loader, DataPartitioner

logger = logging.getLogger(__name__)


class NodeManager:
    """
    Manages the lifecycle of DFL nodes
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize node manager

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.nodes: List[BaseNode] = []
        self.test_loader: Optional[DataLoader] = None

    def create_nodes(self) -> List[BaseNode]:
        """
        Create and initialize all DFL nodes

        Returns:
            List of initialized nodes
        """
        logger.info("Creating DFL nodes...")

        # Load and partition data
        data_loader = CIFAR10Loader(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers']
        )

        train_dataset, test_dataset = data_loader.load_data()
        self.test_loader = data_loader.create_dataloader(test_dataset, shuffle=False)

        # Partition training data
        partitioner = DataPartitioner(
            dataset=train_dataset,
            num_nodes=self.config['partition']['num_nodes'],
            alpha=self.config['partition']['alpha'],
            seed=self.config['experiment']['seed']
        )

        if self.config['partition']['partition_type'] == 'iid':
            node_datasets = partitioner.partition_iid()
        else:
            node_datasets = partitioner.partition_dirichlet()

        # Create nodes
        nodes = []
        for node_id in range(self.config['partition']['num_nodes']):
            # Create model for this node
            model = create_model(
                num_classes=self.config['model']['num_classes'],
                pretrained=self.config['model']['pretrained']
            )

            # Create dataloader for this node
            node_dataloader = data_loader.create_dataloader(
                node_datasets[node_id], shuffle=True
            )

            # Create node
            node = BaseNode(
                node_id=node_id,
                model=model,
                dataloader=node_dataloader,
                config=self.config['training']
            )

            nodes.append(node)

        self.nodes = nodes
        logger.info(f"Created {len(nodes)} DFL nodes")
        return nodes

    def get_nodes(self) -> List[BaseNode]:
        """Get list of nodes"""
        return self.nodes

    def get_test_loader(self) -> Optional[DataLoader]:
        """Get test data loader"""
        return self.test_loader

    def evaluate_all_nodes(self) -> Dict[int, Dict[str, float]]:
        """
        Evaluate all nodes on test data

        Returns:
            Dictionary mapping node_id to evaluation metrics
        """
        if self.test_loader is None:
            raise ValueError("Test loader not available")

        results = {}
        for node in self.nodes:
            metrics = node.evaluate(self.test_loader)
            results[node.node_id] = metrics

        return results

    def get_node_data_sizes(self) -> Dict[int, int]:
        """
        Get data sizes for all nodes

        Returns:
            Dictionary mapping node_id to data size
        """
        return {node.node_id: node.get_data_size() for node in self.nodes}

    def save_node_models(self, save_dir: str, round_num: int) -> None:
        """
        Save all node models

        Args:
            save_dir: Directory to save models
            round_num: Current round number
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        for node in self.nodes:
            model_path = os.path.join(save_dir, f"node_{node.node_id}_round_{round_num}.pth")
            torch.save(node.model.state_dict(), model_path)

        logger.info(f"Saved models for round {round_num} to {save_dir}")

    def load_node_models(self, save_dir: str, round_num: int) -> None:
        """
        Load all node models

        Args:
            save_dir: Directory containing saved models
            round_num: Round number to load
        """
        import os

        for node in self.nodes:
            model_path = os.path.join(save_dir, f"node_{node.node_id}_round_{round_num}.pth")
            if os.path.exists(model_path):
                node.model.load_state_dict(torch.load(model_path))
            else:
                logger.warning(f"Model file not found: {model_path}")

        logger.info(f"Loaded models for round {round_num} from {save_dir}")