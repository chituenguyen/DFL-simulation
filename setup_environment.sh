#!/bin/bash

################################################################################
# DFL Simulation Environment Setup Script
# This script sets up the complete environment for running experiments
# Usage: bash setup_environment.sh
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "  🚀 DFL Simulation Environment Setup"
echo "================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "\n${GREEN}📁 Project root: $PROJECT_ROOT${NC}\n"

################################################################################
# 1. Create necessary directories
################################################################################
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  📂 Creating directories..."
echo "────────────────────────────────────────────────────────────────────────────────"

DIRS=(
    "data"
    "results"
    "results/plots"
    "results/logs"
    "results/checkpoints"
    "results/pretrained_models"
    "results/fl_checkpoints"
    "results/local_models"
    "results/p2p_checkpoints"
    "results/p2p_local_models"
    "src/data"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  ${GREEN}✓${NC} Created: $dir"
    else
        echo -e "  ${YELLOW}⊙${NC} Exists:  $dir"
    fi
done

################################################################################
# 2. Check if src/data files exist, if not download from template
################################################################################
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  📝 Checking src/data files..."
echo "────────────────────────────────────────────────────────────────────────────────"

# Check if data module files exist
if [ ! -f "src/data/__init__.py" ]; then
    echo -e "  ${RED}✗${NC} src/data/__init__.py missing, creating..."

    cat > src/data/__init__.py << 'EOF'
"""
Data loading and partitioning utilities
"""

from .cifar10_loader import CIFAR10Loader
from .data_partitioner import DataPartitioner

__all__ = ["CIFAR10Loader", "DataPartitioner"]
EOF
    echo -e "  ${GREEN}✓${NC} Created src/data/__init__.py"
fi

# Check cifar10_loader.py
if [ ! -f "src/data/cifar10_loader.py" ]; then
    echo -e "  ${RED}✗${NC} src/data/cifar10_loader.py missing"
    echo -e "  ${YELLOW}⚠${NC}  This file should exist in your repository"
    echo -e "  ${YELLOW}⚠${NC}  Please ensure all source files are committed to git"
else
    echo -e "  ${GREEN}✓${NC} src/data/cifar10_loader.py exists"
fi

# Check data_partitioner.py
if [ ! -f "src/data/data_partitioner.py" ]; then
    echo -e "  ${RED}✗${NC} src/data/data_partitioner.py missing"
    echo -e "  ${YELLOW}⚠${NC}  This file should exist in your repository"
    echo -e "  ${YELLOW}⚠${NC}  Please ensure all source files are committed to git"
else
    echo -e "  ${GREEN}✓${NC} src/data/data_partitioner.py exists"
fi

################################################################################
# 3. Download CIFAR-10 dataset if not exists
################################################################################
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  📦 Checking CIFAR-10 dataset..."
echo "────────────────────────────────────────────────────────────────────────────────"

if [ ! -d "data/cifar-10-batches-py" ] && [ ! -d "data/cifar-10-python.tar.gz" ]; then
    echo -e "  ${YELLOW}⊙${NC} CIFAR-10 not found, will be downloaded on first run"
    echo -e "  ${YELLOW}⊙${NC} Dataset will be auto-downloaded by PyTorch (~170MB)"
else
    echo -e "  ${GREEN}✓${NC} CIFAR-10 dataset exists"
fi

################################################################################
# 4. Verify Python environment
################################################################################
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  🐍 Verifying Python environment..."
echo "────────────────────────────────────────────────────────────────────────────────"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "  ${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Check if required packages are installed
REQUIRED_PACKAGES=("torch" "torchvision" "numpy" "matplotlib" "yaml")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Package installed: $package"
    else
        echo -e "  ${RED}✗${NC} Package missing: $package"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠  Missing packages detected. Install with:${NC}"
    echo -e "   pip install torch torchvision numpy matplotlib pyyaml"
    echo ""
fi

################################################################################
# 5. Test imports
################################################################################
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  🧪 Testing imports..."
echo "────────────────────────────────────────────────────────────────────────────────"

# Test if imports work
python3 << 'PYTHON_TEST'
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, project_root)

errors = []

try:
    from src.node import BaseNode
    print("  ✓ src.node imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import src.node: {e}")
    errors.append("src.node")

try:
    from src.models import create_model
    print("  ✓ src.models imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import src.models: {e}")
    errors.append("src.models")

try:
    from src.data import DataPartitioner
    print("  ✓ src.data imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import src.data: {e}")
    errors.append("src.data")

try:
    from src.aggregation import FedAvgAggregator
    print("  ✓ src.aggregation imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import src.aggregation: {e}")
    errors.append("src.aggregation")

try:
    from src.communication import create_topology
    print("  ✓ src.communication imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import src.communication: {e}")
    errors.append("src.communication")

if errors:
    print(f"\n  ⚠ {len(errors)} import(s) failed: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n  ✓ All imports successful!")
    sys.exit(0)
PYTHON_TEST

IMPORT_STATUS=$?

################################################################################
# 6. Check GPU availability
################################################################################
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  🖥️  Checking GPU availability..."
echo "────────────────────────────────────────────────────────────────────────────────"

python3 << 'GPU_CHECK'
import torch

if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"  ✓ MPS (Apple Silicon) available")
else:
    print(f"  ⊙ GPU not available, using CPU")
    print(f"  ⊙ Training will be slower on CPU")
GPU_CHECK

################################################################################
# 7. Summary
################################################################################
echo ""
echo "================================================================================"
echo "  📊 Setup Summary"
echo "================================================================================"

if [ $IMPORT_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}✅  Environment setup complete!${NC}\n"
    echo "You can now run experiments:"
    echo "  • python run_experiment.py dog_federated_improvement"
    echo "  • python run_experiment.py dog_federated_p2p"
    echo "  • python run_experiment.py single_class_dog_demo"
    echo "  • python run_experiment.py class_based_demo"
    echo ""
else
    echo -e "\n${RED}❌  Setup incomplete - some imports failed${NC}\n"
    echo "Please check:"
    echo "  1. All source files are present in src/"
    echo "  2. Required packages are installed"
    echo "  3. Python path is configured correctly"
    echo ""
    echo "For Colab/remote environments, ensure all files were uploaded:"
    echo "  git clone <your-repo-url>"
    echo "  cd DFL-simulation"
    echo "  bash setup_environment.sh"
    echo ""
fi

echo "================================================================================"
