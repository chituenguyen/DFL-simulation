"""
Data Partitioner for Non-IID Data Distribution
"""

import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPartitioner:
    """
    Partition dataset into Non-IID subsets for federated nodes
    Uses Dirichlet distribution for realistic heterogeneity
    """

    def __init__(self, dataset: Dataset, num_nodes: int,
                 alpha: float = 0.5, seed: int = 42):
        """
        Args:
            dataset: PyTorch dataset to partition
            num_nodes: Number of federated nodes
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.seed = seed
        np.random.seed(seed)

    def partition_dirichlet(self) -> List[Subset]:
        """
        Partition data using Dirichlet distribution
        Returns list of Subsets, one per node
        """
        # Get labels from dataset
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            labels = np.array(self.dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")

        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Initialize partition indices
        node_indices = [[] for _ in range(self.num_nodes)]

        # For each class, distribute samples according to Dirichlet
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(
                np.repeat(self.alpha, self.num_nodes)
            )

            # Distribute samples according to proportions
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_splits = np.split(idx_k, proportions)

            # Assign to nodes
            for node_id, idx in enumerate(idx_splits):
                node_indices[node_id].extend(idx.tolist())

        # Log partition statistics
        self._log_partition_stats(node_indices, labels)

        # Create Subsets
        return [Subset(self.dataset, indices) for indices in node_indices]

    def partition_iid(self) -> List[Subset]:
        """
        Partition data uniformly (IID)
        """
        num_samples = len(self.dataset)
        indices = np.random.permutation(num_samples)
        splits = np.array_split(indices, self.num_nodes)

        return [Subset(self.dataset, split.tolist()) for split in splits]

    def partition_class_based(self, class_assignments: List[List[int]]) -> List[Subset]:
        """
        **NEW METHOD** - Partition data based on specific classes for each node.
        Each node only gets data from assigned classes.
        
        Args:
            class_assignments: List of lists, each inner list contains class indices
                              for corresponding node. 
                              E.g., [[0,1], [2,3], [4,5], [6,7], [8,9]] means:
                              - Node 0 gets classes 0, 1 (airplane, automobile)
                              - Node 1 gets classes 2, 3 (bird, cat)
                              - Node 2 gets classes 4, 5 (deer, dog)
                              - etc.
        
        Returns:
            List of Subsets, one per node
        
        Example:
            For CIFAR-10 with 5 nodes (2 classes each):
            partitioner.partition_class_based([[0,1], [2,3], [4,5], [6,7], [8,9]])
        """
        if len(class_assignments) != self.num_nodes:
            raise ValueError(
                f"class_assignments length ({len(class_assignments)}) "
                f"must match num_nodes ({self.num_nodes})"
            )
        
        # Get labels from dataset
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            labels = np.array(self.dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")
        
        # Initialize partition indices
        node_indices = [[] for _ in range(self.num_nodes)]
        
        # For each node, collect indices of assigned classes
        for node_id, assigned_classes in enumerate(class_assignments):
            for class_id in assigned_classes:
                # Get all indices for this class
                idx_k = np.where(labels == class_id)[0]
                node_indices[node_id].extend(idx_k.tolist())
            
            # Shuffle indices for this node
            np.random.shuffle(node_indices[node_id])
        
        # Log partition statistics with class names
        self._log_class_based_stats(node_indices, labels, class_assignments)
        
        # Create Subsets
        return [Subset(self.dataset, indices) for indices in node_indices]

    def _log_class_based_stats(self, node_indices: List[List[int]], 
                                labels: np.ndarray,
                                class_assignments: List[List[int]]) -> None:
        """Log class-based partition statistics"""
        # CIFAR-10 class names for reference
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        logger.info("=" * 70)
        logger.info("CLASS-BASED Data Partition Statistics")
        logger.info("=" * 70)
        
        for node_id, indices in enumerate(node_indices):
            node_labels = labels[indices]
            class_counts = np.bincount(node_labels, minlength=len(class_names))
            assigned_classes = class_assignments[node_id]
            assigned_names = [class_names[c] for c in assigned_classes]
            
            logger.info(f"\n📦 Node {node_id}:")
            logger.info(f"   Total samples: {len(indices)}")
            logger.info(f"   Assigned classes: {assigned_classes} - {assigned_names}")
            
            # Show distribution for assigned classes only
            logger.info(f"   Class distribution:")
            for class_id in assigned_classes:
                logger.info(f"      Class {class_id} ({class_names[class_id]}): "
                          f"{class_counts[class_id]} samples")
        
        logger.info("\n" + "=" * 70)

    def _log_partition_stats(self, node_indices: List[List[int]],
                            labels: np.ndarray) -> None:
        """Log partition statistics for analysis"""
        logger.info(f"Data Partition Statistics (α={self.alpha}):")
        logger.info("-" * 50)

        for node_id, indices in enumerate(node_indices):
            node_labels = labels[indices]
            class_counts = np.bincount(node_labels, minlength=10)
            logger.info(f"Node {node_id}: {len(indices)} samples")
            logger.info(f"  Class distribution: {class_counts}")