"""
Evaluation metrics for DFL simulation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import pandas as pd


class MetricsTracker:
    """
    Track and compute metrics for DFL experiments
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics_history = []
        self.global_metrics_history = []

    def add_node_metrics(self, round_num: int, node_id: int, metrics: Dict[str, float]) -> None:
        """
        Add metrics for a specific node and round

        Args:
            round_num: Training round number
            node_id: Node ID
            metrics: Dictionary with metrics
        """
        record = {
            'round': round_num,
            'node_id': node_id,
            **metrics
        }
        self.metrics_history.append(record)

    def add_global_metrics(self, round_num: int, metrics: Dict[str, float]) -> None:
        """
        Add global metrics for a specific round

        Args:
            round_num: Training round number
            metrics: Dictionary with global metrics
        """
        record = {
            'round': round_num,
            **metrics
        }
        self.global_metrics_history.append(record)

    def compute_global_metrics(self, node_metrics: List[Dict[str, Any]],
                              data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, float]:
        """
        Compute global metrics from individual node metrics

        Args:
            node_metrics: List of node metrics dictionaries
            data_sizes: Dataset sizes for weighted averaging

        Returns:
            Global metrics dictionary
        """
        if not node_metrics:
            return {}

        global_metrics = {}

        # Get metric names from first node
        metric_names = [key for key in node_metrics[0].keys()
                       if key not in ['node_id', 'round'] and isinstance(node_metrics[0][key], (int, float))]

        for metric_name in metric_names:
            if data_sizes and all('data_size' in m or m.get('node_id') in data_sizes for m in node_metrics):
                # Weighted average
                total_weighted_value = 0
                total_weight = 0

                for node_metric in node_metrics:
                    node_id = node_metric.get('node_id', 0)
                    weight = data_sizes.get(node_id, node_metric.get('data_size', 1))
                    value = node_metric.get(metric_name, 0)

                    total_weighted_value += weight * value
                    total_weight += weight

                if total_weight > 0:
                    global_metrics[f'global_{metric_name}'] = total_weighted_value / total_weight
                else:
                    global_metrics[f'global_{metric_name}'] = 0
            else:
                # Simple average
                values = [m.get(metric_name, 0) for m in node_metrics]
                global_metrics[f'global_{metric_name}'] = np.mean(values)

        # Additional statistics
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in node_metrics]
            global_metrics[f'std_{metric_name}'] = np.std(values)
            global_metrics[f'min_{metric_name}'] = np.min(values)
            global_metrics[f'max_{metric_name}'] = np.max(values)

        return global_metrics

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics history as pandas DataFrame

        Returns:
            DataFrame with all metrics
        """
        return pd.DataFrame(self.metrics_history)

    def get_global_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get global metrics history as pandas DataFrame

        Returns:
            DataFrame with global metrics
        """
        return pd.DataFrame(self.global_metrics_history)

    def get_convergence_metrics(self, metric_name: str = 'test_accuracy') -> Dict[str, float]:
        """
        Compute convergence metrics

        Args:
            metric_name: Name of metric to analyze

        Returns:
            Convergence statistics
        """
        global_df = self.get_global_metrics_dataframe()
        if global_df.empty or f'global_{metric_name}' not in global_df.columns:
            return {}

        values = global_df[f'global_{metric_name}'].values
        rounds = global_df['round'].values

        # Find best value and round
        best_value = np.max(values) if 'accuracy' in metric_name else np.min(values)
        best_round = rounds[np.argmax(values) if 'accuracy' in metric_name else np.argmin(values)]

        # Compute convergence metrics
        final_value = values[-1] if len(values) > 0 else 0
        initial_value = values[0] if len(values) > 0 else 0
        improvement = abs(final_value - initial_value)

        # Stability (variance in last 20% of rounds)
        stability_window = max(1, len(values) // 5)
        recent_values = values[-stability_window:] if len(values) >= stability_window else values
        stability = np.std(recent_values)

        return {
            'best_value': best_value,
            'best_round': int(best_round),
            'final_value': final_value,
            'improvement': improvement,
            'stability': stability
        }

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to CSV file

        Args:
            filepath: Path to save CSV file
        """
        df = self.get_metrics_dataframe()
        df.to_csv(filepath, index=False)

    def export_global_metrics(self, filepath: str) -> None:
        """
        Export global metrics to CSV file

        Args:
            filepath: Path to save CSV file
        """
        df = self.get_global_metrics_dataframe()
        df.to_csv(filepath, index=False)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the experiment

        Returns:
            Dictionary with summary statistics
        """
        df = self.get_metrics_dataframe()
        global_df = self.get_global_metrics_dataframe()

        if df.empty:
            return {}

        summary = {
            'total_rounds': df['round'].max() if 'round' in df.columns else 0,
            'num_nodes': df['node_id'].nunique() if 'node_id' in df.columns else 0,
            'total_records': len(df)
        }

        # Add convergence metrics for key metrics
        key_metrics = ['test_accuracy', 'train_accuracy', 'test_loss', 'train_loss']
        for metric in key_metrics:
            if f'global_{metric}' in global_df.columns:
                convergence_stats = self.get_convergence_metrics(metric)
                summary[f'{metric}_convergence'] = convergence_stats

        return summary