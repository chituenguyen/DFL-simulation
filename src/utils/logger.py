"""
Logging utilities for DFL simulation
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "dfl_simulation",
                level: str = "INFO",
                log_dir: Optional[str] = None,
                console: bool = True) -> logging.Logger:
    """
    Set up logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        console: Whether to add console handler

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_dir is specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"dfl_simulation_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file created: {log_file}")

    return logger


class ExperimentLogger:
    """
    Logger for experiment metrics and results
    """

    def __init__(self, experiment_name: str, log_dir: str):
        """
        Initialize experiment logger

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.logger = setup_logger(f"experiment_{experiment_name}", log_dir=log_dir)

        # Create experiment log file
        self.metrics_file = os.path.join(log_dir, f"metrics_{experiment_name}.csv")
        self._init_metrics_file()

    def _init_metrics_file(self) -> None:
        """Initialize CSV file for metrics"""
        os.makedirs(self.log_dir, exist_ok=True)

        # Write CSV header
        header = "round,node_id,train_loss,train_accuracy,test_loss,test_accuracy,data_size\n"
        with open(self.metrics_file, 'w') as f:
            f.write(header)

        self.logger.info(f"Metrics file initialized: {self.metrics_file}")

    def log_round_start(self, round_num: int) -> None:
        """Log start of training round"""
        self.logger.info(f"=== Round {round_num} Started ===")

    def log_data_loading_start(self) -> None:
        """Log start of data loading phase"""
        self.logger.info("📊 Starting data loading and partitioning...")

    def log_data_loading_complete(self, num_nodes: int, total_samples: int) -> None:
        """Log completion of data loading"""
        self.logger.info(f"✅ Data loading complete: {total_samples} samples distributed across {num_nodes} nodes")

    def log_node_training_start(self, round_num: int, node_id: int) -> None:
        """Log start of node training"""
        self.logger.info(f"🚀 Round {round_num}: Node {node_id} starting local training...")

    def log_node_training_complete(self, round_num: int, node_id: int) -> None:
        """Log completion of node training"""
        self.logger.info(f"✅ Round {round_num}: Node {node_id} training complete")

    def log_aggregation_start(self, round_num: int) -> None:
        """Log start of model aggregation"""
        self.logger.info(f"🔄 Round {round_num}: Starting model aggregation...")

    def log_round_end(self, round_num: int, global_metrics: dict) -> None:
        """Log end of training round with global metrics"""
        self.logger.info(f"=== Round {round_num} Complete ===")
        self.logger.info(f"Global Test Accuracy: {global_metrics.get('test_accuracy', 0):.4f}")
        self.logger.info(f"Global Test Loss: {global_metrics.get('test_loss', 0):.4f}")

    def log_node_metrics(self, round_num: int, node_id: int, metrics: dict) -> None:
        """
        Log metrics for a specific node

        Args:
            round_num: Training round number
            node_id: Node ID
            metrics: Dictionary with metrics
        """
        # Log to file
        self.logger.info(f"Node {node_id} - Round {round_num}: {metrics}")

        # Save to CSV
        row = f"{round_num},{node_id}," \
              f"{metrics.get('train_loss', 0):.6f}," \
              f"{metrics.get('train_accuracy', 0):.6f}," \
              f"{metrics.get('test_loss', 0):.6f}," \
              f"{metrics.get('test_accuracy', 0):.6f}," \
              f"{metrics.get('data_size', 0)}\n"

        with open(self.metrics_file, 'a') as f:
            f.write(row)

    def log_experiment_config(self, config: dict) -> None:
        """Log experiment configuration"""
        self.logger.info("=== Experiment Configuration ===")
        for section, params in config.items():
            self.logger.info(f"[{section}]")
            for key, value in params.items():
                self.logger.info(f"  {key}: {value}")

    def log_aggregation_stats(self, round_num: int, stats: dict) -> None:
        """Log aggregation statistics"""
        self.logger.info(f"Round {round_num} Aggregation Stats: {stats}")

    def get_metrics_file(self) -> str:
        """Get path to metrics CSV file"""
        return self.metrics_file