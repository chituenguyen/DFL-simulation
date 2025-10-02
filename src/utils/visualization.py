"""
Visualization utilities for DFL simulation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os


class Visualizer:
    """
    Create visualizations for DFL experiment results
    """

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn')

        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)

    def plot_training_curves(self, metrics_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> None:
        """
        Plot training curves for all nodes

        Args:
            metrics_df: DataFrame with training metrics
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training Loss
        for node_id in metrics_df['node_id'].unique():
            node_data = metrics_df[metrics_df['node_id'] == node_id]
            ax1.plot(node_data['round'], node_data['train_loss'],
                    label=f'Node {node_id}', alpha=0.7)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss by Node')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training Accuracy
        for node_id in metrics_df['node_id'].unique():
            node_data = metrics_df[metrics_df['node_id'] == node_id]
            ax2.plot(node_data['round'], node_data['train_accuracy'],
                    label=f'Node {node_id}', alpha=0.7)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Training Accuracy')
        ax2.set_title('Training Accuracy by Node')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Test Loss
        for node_id in metrics_df['node_id'].unique():
            node_data = metrics_df[metrics_df['node_id'] == node_id]
            ax3.plot(node_data['round'], node_data['test_loss'],
                    label=f'Node {node_id}', alpha=0.7)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Test Loss')
        ax3.set_title('Test Loss by Node')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Test Accuracy
        for node_id in metrics_df['node_id'].unique():
            node_data = metrics_df[metrics_df['node_id'] == node_id]
            ax4.plot(node_data['round'], node_data['test_accuracy'],
                    label=f'Node {node_id}', alpha=0.7)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Test Accuracy by Node')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_global_convergence(self, global_metrics_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
        """
        Plot global convergence metrics

        Args:
            global_metrics_df: DataFrame with global metrics
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        rounds = global_metrics_df['round']

        # Global metrics with error bars
        metrics_to_plot = [
            ('global_train_loss', 'std_train_loss', 'Global Training Loss'),
            ('global_train_accuracy', 'std_train_accuracy', 'Global Training Accuracy'),
            ('global_test_loss', 'std_test_loss', 'Global Test Loss'),
            ('global_test_accuracy', 'std_test_accuracy', 'Global Test Accuracy')
        ]

        axes = [ax1, ax2, ax3, ax4]

        for i, (mean_col, std_col, title) in enumerate(metrics_to_plot):
            if mean_col in global_metrics_df.columns and std_col in global_metrics_df.columns:
                mean_vals = global_metrics_df[mean_col]
                std_vals = global_metrics_df[std_col]

                axes[i].plot(rounds, mean_vals, linewidth=2, color=self.colors[0])
                axes[i].fill_between(rounds, mean_vals - std_vals, mean_vals + std_vals,
                                   alpha=0.3, color=self.colors[0])

            axes[i].set_xlabel('Round')
            axes[i].set_ylabel(title.split()[-1])
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_data_distribution(self, data_sizes: Dict[int, int],
                              save_path: Optional[str] = None) -> None:
        """
        Plot data distribution across nodes

        Args:
            data_sizes: Dictionary mapping node_id to data size
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        nodes = list(data_sizes.keys())
        sizes = list(data_sizes.values())

        # Bar plot
        bars = ax1.bar(nodes, sizes, color=self.colors[:len(nodes)])
        ax1.set_xlabel('Node ID')
        ax1.set_ylabel('Dataset Size')
        ax1.set_title('Data Distribution Across Nodes')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # Pie chart
        ax2.pie(sizes, labels=[f'Node {i}' for i in nodes], autopct='%1.1f%%',
               colors=self.colors[:len(nodes)])
        ax2.set_title('Data Distribution (Percentage)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_network_topology(self, adjacency_matrix: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot network topology

        Args:
            adjacency_matrix: Network adjacency matrix
            save_path: Path to save the plot
        """
        import networkx as nx

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create graph from adjacency matrix
        G = nx.from_numpy_matrix(adjacency_matrix)

        # Network visualization
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color=self.colors[0],
               node_size=1000, font_size=12, font_weight='bold')
        ax1.set_title('Network Topology')

        # Adjacency matrix heatmap
        sns.heatmap(adjacency_matrix, annot=True, fmt='d', ax=ax2,
                   cmap='Blues', cbar_kws={'label': 'Connection'})
        ax2.set_title('Adjacency Matrix')
        ax2.set_xlabel('Node ID')
        ax2.set_ylabel('Node ID')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_comparison(self, experiments_data: Dict[str, pd.DataFrame],
                       metric: str = 'global_test_accuracy',
                       save_path: Optional[str] = None) -> None:
        """
        Plot comparison between different experiments

        Args:
            experiments_data: Dictionary mapping experiment names to DataFrames
            metric: Metric to compare
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)

        for i, (exp_name, df) in enumerate(experiments_data.items()):
            if metric in df.columns:
                plt.plot(df['round'], df[metric], label=exp_name,
                        linewidth=2, color=self.colors[i % len(self.colors)])

        plt.xlabel('Round')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_experiment_report(self, metrics_df: pd.DataFrame,
                                global_metrics_df: pd.DataFrame,
                                config: Dict[str, Any],
                                save_dir: str) -> None:
        """
        Create a comprehensive experiment report

        Args:
            metrics_df: Node metrics DataFrame
            global_metrics_df: Global metrics DataFrame
            config: Experiment configuration
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)

        # Training curves
        self.plot_training_curves(
            metrics_df,
            save_path=os.path.join(save_dir, 'training_curves.png')
        )

        # Global convergence
        self.plot_global_convergence(
            global_metrics_df,
            save_path=os.path.join(save_dir, 'global_convergence.png')
        )

        print(f"Experiment report saved to {save_dir}")