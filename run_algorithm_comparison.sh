#!/bin/bash

# Algorithm Comparison Test Runner
# This script runs the algorithm comparison framework to compare multiple
# optimization algorithms on standard test configurations.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "ALGORITHM COMPARISON TEST SUITE"
echo "================================================================================"
echo ""
echo "This script will compare multiple optimization algorithms on test configurations."
echo "Results will be saved to outputs/alg_comparison_results.json"
echo ""
echo "Starting algorithm comparison..."
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run algorithm comparison
# This compares: Greedy, Conservative, Genetic Algorithm, etc.
cd src
python compare_algorithms.py

# Note: Outputs are saved to src/outputs/ by default
# Copy to project root outputs/ if needed
if [ -f "outputs/alg_comparison_results.json" ]; then
    cp outputs/alg_comparison_results.json ../outputs/ 2>/dev/null || true
fi

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Algorithm comparison completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: src/outputs/alg_comparison_results.json"
    if [ -f "outputs/alg_comparison_results.json" ]; then
        echo "  (also copied to: outputs/alg_comparison_results.json)"
    fi
    echo ""
    echo "To view results:"
    echo "  cat src/outputs/alg_comparison_results.json | python -m json.tool"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "✗ Algorithm comparison failed!"
    echo "================================================================================"
    exit 1
fi
