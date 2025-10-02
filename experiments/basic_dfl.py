"""
Basic DFL Experiment Implementation
"""

import torch
import yaml
import os
import sys
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.node import NodeManager
from src.communication.protocol import SimulatedP2PProtocol
from src.communication.topology import create_topology
from src.aggregation import FedAvgAggregator
from src.utils import setup_logger, MetricsTracker, Visualizer, ExperimentLogger


class BasicDFL:
    """
    Basic Decentralized Federated Learning experiment
    """

    def __init__(self, config_path: str):
        """
        Initialize DFL experiment

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.logger = setup_logger(
            level=self.config['logging']['level'],
            log_dir=self.config['logging']['save_dir']
        )

        # Initialize experiment logger
        self.exp_logger = ExperimentLogger(
            self.config['experiment']['name'],
            self.config['logging']['save_dir']
        )

        # Setup device
        self.device = torch.device(self.config['experiment']['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, using CPU")
            self.device = torch.device('cpu')

        # Set random seeds
        torch.manual_seed(self.config['experiment']['seed'])
        torch.cuda.manual_seed_all(self.config['experiment']['seed'])

        # Initialize components
        self.node_manager = NodeManager(self.config)
        self.metrics_tracker = MetricsTracker()
        self.visualizer = Visualizer()

        # Initialize network topology
        self.topology = create_topology(
            topology_type=self.config['topology']['type'],
            num_nodes=self.config['partition']['num_nodes'],
            **self.config['topology']
        )

        # Initialize aggregator
        self.aggregator = FedAvgAggregator(
            weighted=self.config['aggregation']['weighted']
        )

        self.logger.info("DFL experiment initialized successfully")

    def run(self) -> None:
        """Run the complete DFL experiment"""
        self.logger.info("Starting DFL experiment")
        self.exp_logger.log_experiment_config(self.config)

        # Create nodes
        nodes = self.node_manager.create_nodes()
        test_loader = self.node_manager.get_test_loader()
        data_sizes = self.node_manager.get_node_data_sizes()

        # Create communication protocols for each node
        protocols = {}
        topology_neighbors = self.topology.get_all_neighbors()

        for node in nodes:
            protocols[node.node_id] = SimulatedP2PProtocol(
                node.node_id,
                topology_neighbors[node.node_id]
            )

        # Log topology information
        topology_info = self.topology.get_topology_info()
        self.logger.info(f"Network topology: {topology_info}")

        # Training rounds
        num_rounds = self.config['training']['num_rounds']
        local_epochs = self.config['training']['local_epochs']
        save_frequency = self.config['checkpoint']['save_frequency']

        for round_num in range(1, num_rounds + 1):
            self.exp_logger.log_round_start(round_num)

            # 1. Local training
            round_metrics = []
            for node in nodes:
                # Perform local training
                train_metrics = node.local_train(local_epochs)

                # Evaluate on test set
                test_metrics = node.evaluate(test_loader)

                # Combine metrics
                node_metrics = {
                    **train_metrics,
                    **test_metrics,
                    'node_id': node.node_id,
                    'data_size': node.get_data_size()
                }

                round_metrics.append(node_metrics)

                # Log node metrics
                self.exp_logger.log_node_metrics(round_num, node.node_id, node_metrics)
                self.metrics_tracker.add_node_metrics(round_num, node.node_id, node_metrics)

            # 2. Communication phase
            model_updates = []
            for node in nodes:
                # Get model parameters
                model_params = node.get_model_parameters()
                model_updates.append((node.node_id, model_params))

                # Broadcast to neighbors (simulated)
                protocol = protocols[node.node_id]
                protocol.broadcast_model(model_params)

            # 3. Aggregation phase
            aggregated_params = self.aggregator.aggregate(model_updates, data_sizes)
            aggregation_stats = self.aggregator.compute_aggregation_stats(model_updates, data_sizes)
            self.exp_logger.log_aggregation_stats(round_num, aggregation_stats)

            # 4. Update all nodes with aggregated parameters
            for node in nodes:
                node.set_model_parameters(aggregated_params)

            # 5. Compute global metrics
            global_metrics = self.metrics_tracker.compute_global_metrics(round_metrics, data_sizes)
            self.metrics_tracker.add_global_metrics(round_num, global_metrics)
            self.exp_logger.log_round_end(round_num, global_metrics)

            # 6. Save checkpoints
            if round_num % save_frequency == 0:
                self.node_manager.save_node_models(
                    self.config['checkpoint']['save_dir'],
                    round_num
                )

            # 7. Clear communication buffers
            SimulatedP2PProtocol.reset_global_buffer()

        # Generate final report
        self._generate_final_report()

        self.logger.info("DFL experiment completed successfully")

    def _generate_final_report(self) -> None:
        """Generate final experiment report"""
        self.logger.info("Generating final experiment report")

        # Get metrics DataFrames
        metrics_df = self.metrics_tracker.get_metrics_dataframe()
        global_metrics_df = self.metrics_tracker.get_global_metrics_dataframe()

        # Create results directory
        results_dir = os.path.join("results", "plots", self.config['experiment']['name'])
        os.makedirs(results_dir, exist_ok=True)

        # Generate visualizations
        self.visualizer.create_experiment_report(
            metrics_df,
            global_metrics_df,
            self.config,
            results_dir
        )

        # Export metrics to CSV
        metrics_csv = os.path.join("results", f"{self.config['experiment']['name']}_metrics.csv")
        global_csv = os.path.join("results", f"{self.config['experiment']['name']}_global_metrics.csv")

        self.metrics_tracker.export_metrics(metrics_csv)
        self.metrics_tracker.export_global_metrics(global_csv)

        # Print summary statistics
        summary_stats = self.metrics_tracker.get_summary_stats()
        self.logger.info("=== Experiment Summary ===")
        for key, value in summary_stats.items():
            self.logger.info(f"{key}: {value}")


def load_config(config_path: str = None) -> str:
    """Load configuration file path"""
    if config_path is None:
        # Default config path
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.yaml'
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Basic DFL Experiment")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        config_path = load_config(args.config)
        experiment = BasicDFL(config_path)
        experiment.run()
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)