"""
Enhanced DFL Training Implementation
Complete orchestration of decentralized federated learning
"""

import torch
import yaml
import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from node import BaseNode, NodeManager
from communication.dfl_protocol import DFLCommunicationProtocol
from communication.topology import create_topology
from aggregation import FedAvgAggregator
from utils import setup_logger, MetricsTracker, Visualizer, ExperimentLogger

logger = logging.getLogger(__name__)


class EnhancedDFLTrainer:
    """
    Enhanced DFL Trainer with realistic P2P communication
    and comprehensive experiment management
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize enhanced DFL trainer"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.logger = setup_logger(
            name="enhanced_dfl",
            level=self.config['logging']['level'],
            log_dir=self.config['logging']['save_dir']
        )

        # Initialize experiment logger
        self.exp_logger = ExperimentLogger(
            f"{self.config['experiment']['name']}_enhanced",
            self.config['logging']['save_dir']
        )

        # Setup device
        self.device = self._get_device()

        # Set random seeds for reproducibility
        self._set_seeds()

        # Initialize components
        self.node_manager = NodeManager(self.config)
        self.metrics_tracker = MetricsTracker()
        self.visualizer = Visualizer()

        # DFL specific components
        self.nodes = []
        self.communication_protocols = {}
        self.topology = None
        self.aggregator = None

        self.logger.info("Enhanced DFL trainer initialized successfully")

    def _get_device(self) -> str:
        """Get optimal computation device"""
        device_config = self.config['experiment']['device']

        if device_config == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        else:
            self.logger.warning(f"Requested device '{device_config}' not available, using CPU")
            return 'cpu'

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        seed = self.config['experiment']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.logger.info(f"Random seeds set to {seed}")

    def setup_experiment(self) -> None:
        """Setup complete DFL experiment"""
        self.logger.info("Setting up DFL experiment...")

        # 1. Create and initialize nodes
        self._setup_nodes()

        # 2. Setup network topology
        self._setup_topology()

        # 3. Initialize communication protocols
        self._setup_communication()

        # 4. Initialize aggregation algorithm
        self._setup_aggregation()

        self.logger.info("✅ DFL experiment setup complete")

    def _setup_nodes(self) -> None:
        """Setup DFL nodes with data partitioning"""
        self.logger.info("Creating DFL nodes...")

        # Create nodes using NodeManager
        self.nodes = self.node_manager.create_nodes()

        self.logger.info(f"✅ Created {len(self.nodes)} DFL nodes")

        # Log data distribution
        data_sizes = self.node_manager.get_node_data_sizes()
        total_samples = sum(data_sizes.values())

        self.logger.info("Data distribution:")
        for node_id, size in data_sizes.items():
            percentage = (size / total_samples) * 100
            self.logger.info(f"  Node {node_id}: {size} samples ({percentage:.1f}%)")

    def _setup_topology(self) -> None:
        """Setup network topology"""
        self.logger.info("Setting up network topology...")

        self.topology = create_topology(
            topology_type=self.config['topology']['type'],
            num_nodes=len(self.nodes),
            **self.config['topology']
        )

        # Assign neighbors to nodes
        neighbor_mapping = self.topology.get_all_neighbors()
        for node in self.nodes:
            neighbors = neighbor_mapping[node.node_id]
            node.set_neighbors(neighbors)

        # Log topology info
        topology_info = self.topology.get_topology_info()
        self.logger.info(f"✅ Network topology ({self.config['topology']['type']}):")
        self.logger.info(f"  Average degree: {topology_info['average_degree']:.1f}")
        self.logger.info(f"  Total edges: {topology_info['total_edges']}")

    def _setup_communication(self) -> None:
        \"\"\"Setup P2P communication protocols\"\"\"
        self.logger.info("Initializing communication protocols...")

        # Communication parameters
        latency_ms = self.config.get('communication', {}).get('latency_ms', 10.0)
        bandwidth_mbps = self.config.get('communication', {}).get('bandwidth_mbps', 100.0)

        # Create communication protocol for each node
        for node in self.nodes:
            protocol = DFLCommunicationProtocol(
                node_id=node.node_id,
                neighbors=node.get_neighbors(),
                latency_ms=latency_ms,
                bandwidth_mbps=bandwidth_mbps
            )
            self.communication_protocols[node.node_id] = protocol

        self.logger.info(f"✅ Communication protocols initialized for {len(self.nodes)} nodes")

    def _setup_aggregation(self) -> None:
        \"\"\"Setup aggregation algorithm\"\"\"
        self.logger.info("Setting up aggregation algorithm...")

        # Enhanced aggregator with momentum
        momentum = self.config.get('aggregation', {}).get('momentum', 0.0)
        adaptive = self.config.get('aggregation', {}).get('adaptive_weighting', False)

        self.aggregator = FedAvgAggregator(
            weighted=self.config['aggregation']['weighted'],
            momentum=momentum,
            adaptive_weighting=adaptive
        )

        self.logger.info(f"✅ FedAvg aggregator initialized (momentum={momentum})")

    def train(self) -> None:
        \"\"\"Execute complete DFL training\"\"\"
        self.logger.info("Starting Enhanced DFL Training")
        self.exp_logger.log_experiment_config(self.config)

        # Training parameters
        num_rounds = self.config['training']['num_rounds']
        local_epochs = self.config['training']['local_epochs']
        save_frequency = self.config['checkpoint']['save_frequency']

        # Get test loader for evaluation
        test_loader = self.node_manager.get_test_loader()
        data_sizes = self.node_manager.get_node_data_sizes()

        # Training loop
        for round_num in range(num_rounds):
            self.exp_logger.log_round_start(round_num + 1)
            round_start_time = time.time()

            # Phase 1: Local Training
            self._local_training_phase(round_num, local_epochs)

            # Phase 2: P2P Communication
            self._communication_phase(round_num)

            # Phase 3: Model Aggregation
            self._aggregation_phase(round_num)

            # Phase 4: Evaluation
            round_metrics = self._evaluation_phase(round_num, test_loader)

            # Phase 5: Global Metrics and Logging
            global_metrics = self._compute_global_metrics(round_metrics, data_sizes)
            self.metrics_tracker.add_global_metrics(round_num + 1, global_metrics)
            self.exp_logger.log_round_end(round_num + 1, global_metrics)

            # Phase 6: Checkpointing
            if (round_num + 1) % save_frequency == 0:
                self._save_checkpoints(round_num + 1)

            # Round timing
            round_time = time.time() - round_start_time
            self.logger.info(f"Round {round_num + 1} completed in {round_time:.2f} seconds")

        # Generate final report
        self._generate_final_report()

        self.logger.info("Enhanced DFL training completed successfully!")

    def _local_training_phase(self, round_num: int, local_epochs: int) -> None:
        \"\"\"Execute local training phase\"\"\"
        self.logger.info(f"Round {round_num + 1}: Local Training Phase")

        for node in tqdm(self.nodes, desc="Local training", leave=False):
            train_metrics = node.local_train(local_epochs)

            # Log node metrics
            self.exp_logger.log_node_metrics(round_num + 1, node.node_id, train_metrics)
            self.metrics_tracker.add_node_metrics(round_num + 1, node.node_id, train_metrics)

    def _communication_phase(self, round_num: int) -> None:
        \"\"\"Execute P2P communication phase\"\"\"
        self.logger.info(f"Round {round_num + 1}: P2P Communication Phase")

        # Each node broadcasts its model to neighbors
        for node in tqdm(self.nodes, desc="Broadcasting models", leave=False):
            model_update = node.prepare_model_for_sharing()
            protocol = self.communication_protocols[node.node_id]
            protocol.broadcast_to_neighbors({'model_update': model_update})

        # Each node receives updates from neighbors
        for node in self.nodes:
            protocol = self.communication_protocols[node.node_id]
            messages = protocol.receive_messages()

            # Process received messages
            neighbor_updates = {}
            for message in messages:
                sender_id = message['sender_id']
                if 'model_update' in message['content']:
                    neighbor_updates[sender_id] = message['content']['model_update']

            # Update node with neighbor models
            node.update_from_neighbors(neighbor_updates)

    def _aggregation_phase(self, round_num: int) -> None:
        \"\"\"Execute model aggregation phase\"\"\"
        self.logger.info(f"Round {round_num + 1}: Model Aggregation Phase")

        # Get communication stats
        comm_stats = DFLCommunicationProtocol.get_global_stats()
        self.logger.info(f"Communication stats: {comm_stats}")

        # Reset communication buffer for next round
        DFLCommunicationProtocol.reset_global_buffer()

    def _evaluation_phase(self, round_num: int, test_loader) -> List[Dict[str, Any]]:
        \"\"\"Execute evaluation phase\"\"\"
        self.logger.info(f"Round {round_num + 1}: Evaluation Phase")

        round_metrics = []
        for node in tqdm(self.nodes, desc="Evaluating nodes", leave=False):
            test_metrics = node.evaluate(test_loader)
            train_metrics = {
                'train_loss': node.history['train_loss'][-1] if node.history['train_loss'] else 0,
                'train_accuracy': node.history['train_accuracy'][-1] if node.history['train_accuracy'] else 0
            }

            # Combine metrics
            node_metrics = {
                **train_metrics,
                **test_metrics,
                'node_id': node.node_id,
                'data_size': node.get_data_size(),
                'neighbors': len(node.get_neighbors())
            }

            round_metrics.append(node_metrics)

        return round_metrics

    def _compute_global_metrics(self, round_metrics: List[Dict], data_sizes: Dict) -> Dict[str, float]:
        \"\"\"Compute global metrics from node metrics\"\"\"
        return self.metrics_tracker.compute_global_metrics(round_metrics, data_sizes)

    def _save_checkpoints(self, round_num: int) -> None:
        \"\"\"Save model checkpoints\"\"\"
        self.node_manager.save_node_models(
            self.config['checkpoint']['save_dir'],
            round_num
        )

    def _generate_final_report(self) -> None:
        \"\"\"Generate comprehensive final report\"\"\"
        self.logger.info("Generating final experiment report...")

        # Get metrics DataFrames
        metrics_df = self.metrics_tracker.get_metrics_dataframe()
        global_metrics_df = self.metrics_tracker.get_global_metrics_dataframe()

        # Create results directory
        results_dir = Path("results") / "plots" / f"{self.config['experiment']['name']}_enhanced"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        self.visualizer.create_experiment_report(
            metrics_df,
            global_metrics_df,
            self.config,
            str(results_dir)
        )

        # Export metrics
        experiment_name = f"{self.config['experiment']['name']}_enhanced"
        metrics_csv = f"results/{experiment_name}_metrics.csv"
        global_csv = f"results/{experiment_name}_global_metrics.csv"

        self.metrics_tracker.export_metrics(metrics_csv)
        self.metrics_tracker.export_global_metrics(global_csv)

        # Print summary
        summary_stats = self.metrics_tracker.get_summary_stats()
        self.logger.info("=== Final Experiment Summary ===")
        for key, value in summary_stats.items():
            self.logger.info(f"{key}: {value}")

        self.logger.info(f"Results saved to: {results_dir}")


def main():
    \"\"\"Main entry point\"\"\"
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced DFL Training")
    parser.add_argument("--config", "-c", default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", choices=['cuda', 'cpu', 'mps'],
                       help="Override device setting")

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = EnhancedDFLTrainer(args.config)

        # Override device if specified
        if args.device:
            trainer.config['experiment']['device'] = args.device
            trainer.device = trainer._get_device()

        # Setup and run experiment
        trainer.setup_experiment()
        trainer.train()

        print("✅ Enhanced DFL training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())