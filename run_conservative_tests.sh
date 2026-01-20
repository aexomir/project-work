#!/bin/bash

# Conservative Optimizer Test Runner
# This script runs the Conservative Optimizer on all test cases from generate_test_cases.py
# and generates comprehensive results and statistics.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "CONSERVATIVE OPTIMIZER TEST SUITE"
echo "================================================================================"
echo ""
echo "This script will run the Conservative Optimizer on all test cases."
echo "Results will be saved to outputs/conservative_comparison_results.json"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run conservative optimizer comparison on all test cases
cd src
python run_conservative_comparison.py

# Note: Outputs are saved to src/outputs/ by default
# Copy to project root outputs/ if needed
if [ -f "outputs/conservative_comparison_results.json" ]; then
    cp outputs/conservative_comparison_results.json ../outputs/ 2>/dev/null || true
fi

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Conservative Optimizer tests completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: src/outputs/conservative_comparison_results.json"
    if [ -f "outputs/conservative_comparison_results.json" ]; then
        echo "  (also copied to: outputs/conservative_comparison_results.json)"
    fi
    echo ""
    echo "Summary of results:"
    echo "  - Test cases run: All test cases from generate_test_cases.py"
    echo "  - Algorithm: Conservative Optimizer"
    echo "  - Output: src/outputs/conservative_comparison_results.json"
    echo ""
    echo "To view results:"
    echo "  cat src/outputs/conservative_comparison_results.json | python -m json.tool"
    echo ""
    echo "To view summary statistics:"
    echo "  python -c \"import json; data=json.load(open('src/outputs/conservative_comparison_results.json')); print('Success Rate:', data['summary']['success_rate'], '%'); print('Avg Improvement:', data['summary']['avg_improvement'], '%')\""
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "✗ Conservative Optimizer tests failed!"
    echo "================================================================================"
    exit 1
fi
