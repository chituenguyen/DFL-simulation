"""
Experiment Runner for DFL Simulations
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.basic_dfl import BasicDFL


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="DFL Experiment Runner")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="basic",
        choices=["basic"],
        help="Type of experiment to run"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (overrides config)"
    )

    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=None,
        help="Number of training rounds (overrides config)"
    )

    parser.add_argument(
        "--nodes", "-n",
        type=int,
        default=None,
        help="Number of nodes (overrides config)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        print("Available configuration files:")
        config_dir = "config"
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith(".yaml") or file.endswith(".yml"):
                    print(f"  {os.path.join(config_dir, file)}")
        sys.exit(1)

    # Load and modify configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply command line overrides
    if args.device:
        config['experiment']['device'] = args.device

    if args.rounds:
        config['training']['num_rounds'] = args.rounds

    if args.nodes:
        config['partition']['num_nodes'] = args.nodes

    if args.seed:
        config['experiment']['seed'] = args.seed

    if args.verbose:
        config['logging']['level'] = 'DEBUG'

    # Validate configuration
    if not validate_config(config):
        sys.exit(1)

    # Create results directories
    create_directories(config)

    # Print experiment information
    print("=" * 60)
    print("DFL Basic Simulation")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Nodes: {config['partition']['num_nodes']}")
    print(f"Rounds: {config['training']['num_rounds']}")
    print(f"Topology: {config['topology']['type']}")
    print(f"Aggregation: {config['aggregation']['algorithm']}")
    print(f"Data Partition: {config['partition']['partition_type']}")
    print("=" * 60)

    # Run experiment
    try:
        if args.experiment == "basic":
            # Save modified config temporarily
            temp_config_path = "temp_config.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Run experiment
            experiment = BasicDFL(temp_config_path)
            experiment.run()

            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        else:
            print(f"Error: Unknown experiment type '{args.experiment}'")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print(f"Results saved to: results/")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate experiment configuration

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_sections = ['experiment', 'model', 'data', 'partition', 'training', 'topology', 'aggregation']

    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in configuration")
            return False

    # Validate device
    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, will use CPU")

    # Validate positive integers
    positive_int_params = [
        ('partition', 'num_nodes'),
        ('training', 'num_rounds'),
        ('training', 'local_epochs'),
        ('model', 'num_classes')
    ]

    for section, param in positive_int_params:
        value = config[section][param]
        if not isinstance(value, int) or value <= 0:
            print(f"Error: {section}.{param} must be a positive integer, got {value}")
            return False

    # Validate probability values
    prob_params = [
        ('partition', 'alpha'),
        ('training', 'learning_rate'),
        ('training', 'momentum'),
        ('training', 'weight_decay')
    ]

    for section, param in prob_params:
        if param in config[section]:
            value = config[section][param]
            if not isinstance(value, (int, float)) or value < 0:
                print(f"Error: {section}.{param} must be non-negative, got {value}")
                return False

    return True


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for experiment

    Args:
        config: Configuration dictionary
    """
    directories = [
        "results",
        "results/logs",
        "results/models",
        "results/plots",
        config['logging']['save_dir'],
        config['checkpoint']['save_dir']
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    main()