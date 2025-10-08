"""
Setup script for Google Colab
Run this before running experiments
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"✅ Added to Python path: {project_root}")

# Verify imports
try:
    from src.node import BaseNode
    from src.models import create_model
    from src.data import DataPartitioner, CIFAR10Loader
    from src.aggregation import FedAvgAggregator
    from src.communication import create_topology, SimulatedP2PProtocol

    print("✅ All imports successful!")
    print("\n📦 Available modules:")
    print("  - src.node (BaseNode, NodeManager)")
    print("  - src.models (create_model)")
    print("  - src.data (DataPartitioner, CIFAR10Loader)")
    print("  - src.aggregation (FedAvgAggregator)")
    print("  - src.communication (create_topology, SimulatedP2PProtocol)")
    print("\n🚀 Ready to run experiments!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔍 Debugging info:")
    print(f"   sys.path: {sys.path}")
    print(f"   Current dir: {os.getcwd()}")
    print(f"   Project files: {os.listdir(project_root)}")
    sys.exit(1)
