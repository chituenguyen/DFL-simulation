"""
FedAvg Aggregation Algorithm
"""

import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FedAvgAggregator:
    """
    Enhanced Federated Averaging (FedAvg) aggregation algorithm
    Supports weighted averaging, momentum, and adaptive learning rates
    """

    def __init__(self, weighted: bool = True, momentum: float = 0.0,
                 adaptive_weighting: bool = False):
        """
        Initialize FedAvg aggregator

        Args:
            weighted: Whether to weight by dataset size
            momentum: Momentum factor for aggregation (0.0 = no momentum)
            adaptive_weighting: Whether to use adaptive weighting based on performance
        """
        self.weighted = weighted
        self.momentum = momentum
        self.adaptive_weighting = adaptive_weighting
        self.previous_global_params = None
        self.node_performance_history = {}

    def aggregate(self, model_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
                  data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate model parameters using FedAvg

        Args:
            model_updates: List of (node_id, model_parameters) tuples
            data_sizes: Dictionary mapping node_id to dataset size (for weighted averaging)

        Returns:
            Aggregated model parameters
        """
        if not model_updates:
            raise ValueError("No model updates provided for aggregation")

        # Extract parameters from first model to get structure
        _, first_params = model_updates[0]
        aggregated_params = {}

        # Initialize aggregated parameters
        for param_name in first_params.keys():
            aggregated_params[param_name] = torch.zeros_like(first_params[param_name])

        # Calculate weights
        if self.weighted and data_sizes is not None:
            total_data = sum(data_sizes.get(node_id, 1) for node_id, _ in model_updates)
            weights = {node_id: data_sizes.get(node_id, 1) / total_data
                      for node_id, _ in model_updates}
        else:
            # Uniform weights
            num_models = len(model_updates)
            weights = {node_id: 1.0 / num_models for node_id, _ in model_updates}

        # Aggregate parameters
        for node_id, model_params in model_updates:
            weight = weights[node_id]

            for param_name, param_value in model_params.items():
                if param_name in aggregated_params:
                    aggregated_params[param_name] += weight * param_value
                else:
                    logger.warning(f"Parameter {param_name} not found in base model")

        # Apply momentum if enabled
        if self.momentum > 0.0 and self.previous_global_params is not None:
            for param_name in aggregated_params.keys():
                if param_name in self.previous_global_params:
                    aggregated_params[param_name] = (
                        (1 - self.momentum) * aggregated_params[param_name] +
                        self.momentum * self.previous_global_params[param_name]
                    )

        # Store current parameters for next round
        self.previous_global_params = {
            name: param.clone() for name, param in aggregated_params.items()
        }

        logger.info(f"Aggregated {len(model_updates)} models "
                   f"({'weighted' if self.weighted else 'uniform'}, momentum={self.momentum})")

        return aggregated_params

    def aggregate_gradients(self, gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
                           data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients instead of model parameters

        Args:
            gradients: List of (node_id, gradients) tuples
            data_sizes: Dictionary mapping node_id to dataset size

        Returns:
            Aggregated gradients
        """
        return self.aggregate(gradients, data_sizes)

    def compute_model_difference(self, global_model: Dict[str, torch.Tensor],
                                local_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute difference between local and global model

        Args:
            global_model: Global model parameters
            local_model: Local model parameters

        Returns:
            Parameter differences (local - global)
        """
        differences = {}

        for param_name in global_model.keys():
            if param_name in local_model:
                differences[param_name] = local_model[param_name] - global_model[param_name]
            else:
                logger.warning(f"Parameter {param_name} not found in local model")

        return differences

    def apply_update(self, base_model: Dict[str, torch.Tensor],
                    update: Dict[str, torch.Tensor],
                    learning_rate: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Apply parameter update to base model

        Args:
            base_model: Base model parameters
            update: Parameter updates to apply
            learning_rate: Learning rate for update

        Returns:
            Updated model parameters
        """
        updated_model = {}

        for param_name in base_model.keys():
            if param_name in update:
                updated_model[param_name] = base_model[param_name] + learning_rate * update[param_name]
            else:
                updated_model[param_name] = base_model[param_name].clone()

        return updated_model

    def compute_aggregation_stats(self, model_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
                                 data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, float]:
        """
        Compute statistics about the aggregation

        Args:
            model_updates: List of model updates
            data_sizes: Dataset sizes for each node

        Returns:
            Aggregation statistics
        """
        num_participants = len(model_updates)
        total_parameters = 0

        if model_updates:
            _, first_params = model_updates[0]
            total_parameters = sum(param.numel() for param in first_params.values())

        total_data = 0
        if data_sizes:
            total_data = sum(data_sizes.get(node_id, 0) for node_id, _ in model_updates)

        stats = {
            'num_participants': num_participants,
            'total_parameters': total_parameters,
            'total_data_samples': total_data,
            'weighted_aggregation': self.weighted
        }

        # Add aggregation method info
        stats.update({
            'aggregation_method': 'fedavg',
            'momentum': self.momentum,
            'adaptive_weighting': self.adaptive_weighting
        })

        return stats