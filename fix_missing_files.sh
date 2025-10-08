#!/bin/bash

################################################################################
# Fix Missing Files Script
# Checks for and reports missing source files
# Usage: bash fix_missing_files.sh
################################################################################

set -e

echo "================================================================================"
echo "  🔍 Checking for missing source files..."
echo "================================================================================"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

MISSING=0

# Check critical files
CRITICAL_FILES=(
    "src/__init__.py"
    "src/data/__init__.py"
    "src/data/cifar10_loader.py"
    "src/data/data_partitioner.py"
    "src/node/__init__.py"
    "src/node/base_node.py"
    "src/node/node_manager.py"
    "src/models/__init__.py"
    "src/models/resnet.py"
    "src/aggregation/__init__.py"
    "src/aggregation/fedavg.py"
    "src/communication/__init__.py"
    "src/communication/topology.py"
    "src/communication/protocol.py"
    "src/utils/__init__.py"
)

echo ""
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file ${RED}(MISSING)${NC}"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}✅  All critical files present!${NC}"
else
    echo -e "${RED}❌  $MISSING file(s) missing!${NC}"
    echo ""
    echo "To fix:"
    echo "  1. Check if files are in .gitignore"
    echo "  2. Force add missing files:"
    echo "     git add -f src/data/*.py"
    echo "     git commit -m 'Add missing source files'"
    echo "     git push"
    echo ""
    echo "  3. Or manually copy files to correct location"
fi

echo "================================================================================"

exit $MISSING
