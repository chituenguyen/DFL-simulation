#!/usr/bin/env python3
"""
Wrapper script to run experiments with proper Python path setup
Usage: python run_experiment.py <experiment_name>
Example: python run_experiment.py dog_federated_improvement
"""

import sys
import os
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Verify imports work
try:
    from src.node import BaseNode
    from src.data import DataPartitioner
    print(f"✅ Imports verified from: {project_root}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_name>")
        print("\nAvailable experiments:")
        exp_dir = os.path.join(project_root, "experiments")
        for f in os.listdir(exp_dir):
            if f.endswith('.py') and not f.startswith('__'):
                print(f"  - {f.replace('.py', '')}")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if not experiment_name.endswith('.py'):
        experiment_name += '.py'

    experiment_path = os.path.join(project_root, 'experiments', experiment_name)

    if not os.path.exists(experiment_path):
        print(f"❌ Experiment not found: {experiment_path}")
        sys.exit(1)

    print(f"🚀 Running experiment: {experiment_name}")
    print(f"   Path: {experiment_path}")
    print(f"   Working dir: {project_root}")
    print("=" * 60)

    # Run experiment with proper environment
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root

    result = subprocess.run(
        [sys.executable, experiment_path],
        cwd=project_root,
        env=env
    )

    sys.exit(result.returncode)
