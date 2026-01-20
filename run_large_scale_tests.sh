#!/bin/bash

# Large Scale Conservative Optimizer Test Runner
# This script runs the Conservative Optimizer on large problem sizes
# from 100 to 1000 cities to test scalability and performance.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "LARGE SCALE CONSERVATIVE OPTIMIZER TEST SUITE"
echo "================================================================================"
echo ""
echo "This script will run the Conservative Optimizer on large problem sizes."
echo "Testing from 100 to 1000 cities (increments of 100)."
echo "Results will be saved to outputs/large_scale_results.json"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Create Python script to generate large scale test cases and run them
cd src
python << 'PYTHON_SCRIPT'
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline.baseline_impl import Problem
from algorithms.conservative_optimizer import ConservativeOptimizer
from utils.evaluator import validate_solution, evaluate_solution

def generate_large_scale_test_cases():
    """Generate test cases for large scale testing (100-1000 cities)"""
    test_cases = []
    
    # Test sizes: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    city_sizes = list(range(100, 1001, 100))
    
    # For each city size, test with different alpha/beta combinations
    for num_cities in city_sizes:
        # Test 1: α=1.0, β=1.5 (moderate penalty)
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 1.0,
            'beta': 1.5,
            'density': 0.5,
            'seed': 42
        })
        # Test 2: α=1.0, β=2.0 (high penalty)
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 1.0,
            'beta': 2.0,
            'density': 0.5,
            'seed': 42
        })
        # Test 3: α=2.0, β=2.0 (very high penalty)
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 2.0,
            'beta': 2.0,
            'density': 0.5,
            'seed': 42
        })
    
    # Add IDs
    for i, test_case in enumerate(test_cases, 1):
        test_case['id'] = i
    
    return test_cases

def run_large_scale_tests(test_cases_list, output_dir="../outputs"):
    """Run Conservative optimizer on large scale test cases"""
    results = {
        'summary': {},
        'detailed': []
    }
    
    print("=" * 80)
    print("LARGE SCALE CONSERVATIVE OPTIMIZER TESTING")
    print("=" * 80)
    print(f"Running {len(test_cases_list)} test cases...")
    print()
    
    for test_idx, test_config in enumerate(test_cases_list, 1):
        test_id = test_config.get('id', test_idx)
        print(f"[{test_idx}/{len(test_cases_list)}] Test Case {test_id}: "
              f"{test_config['num_cities']} cities, "
              f"α={test_config['alpha']:.1f}, β={test_config['beta']:.1f}, "
              f"density={test_config['density']:.1f}, seed={test_config['seed']}")
        
        try:
            # Create problem instance
            print("  Creating problem instance...")
            problem_start = time.time()
            problem = Problem(
                num_cities=test_config['num_cities'],
                alpha=test_config['alpha'],
                beta=test_config['beta'],
                density=test_config['density'],
                seed=test_config['seed']
            )
            problem_time = time.time() - problem_start
            print(f"  Problem created in {problem_time:.2f}s")
            
            # Calculate baseline cost
            print("  Calculating baseline...")
            baseline_start = time.time()
            baseline_cost = problem.baseline()
            baseline_time = time.time() - baseline_start
            print(f"  Baseline: {baseline_cost:,.2f} (calculated in {baseline_time:.2f}s)")
            
            # Run Conservative optimizer
            print("  Running Conservative Optimizer...")
            optimizer = ConservativeOptimizer(
                problem,
                max_iterations=1,
                verbose=False,
                seed=test_config['seed']
            )
            
            solution_start = time.time()
            solution, solution_cost = optimizer.optimize()
            solution_time = time.time() - solution_start
            
            # Validate solution
            print("  Validating solution...")
            is_valid, error_msg = validate_solution(problem, solution)
            
            # Calculate improvement and speedup
            if baseline_cost > 0:
                improvement = ((baseline_cost - solution_cost) / baseline_cost) * 100
                speedup = baseline_cost / solution_cost if solution_cost > 0 else 0
            else:
                improvement = 0.0
                speedup = 0.0
            
            # Store results
            test_result = {
                'id': test_id,
                'config': {
                    'num_cities': test_config['num_cities'],
                    'alpha': test_config['alpha'],
                    'beta': test_config['beta'],
                    'density': test_config['density'],
                    'seed': test_config['seed']
                },
                'baseline_cost': baseline_cost,
                'baseline_time': baseline_time,
                'solution_cost': solution_cost,
                'solution_time': solution_time,
                'problem_time': problem_time,
                'total_time': problem_time + baseline_time + solution_time,
                'improvement_pct': improvement,
                'speedup': speedup,
                'valid': is_valid,
                'error': error_msg if not is_valid else None
            }
            
            results['detailed'].append(test_result)
            
            # Print progress
            status = "✓" if is_valid else "✗"
            print(f"  {status} Solution: {solution_cost:,.2f}, "
                  f"Improvement: {improvement:.2f}%, "
                  f"Time: {solution_time:.2f}s, "
                  f"Total: {test_result['total_time']:.2f}s")
            
            if not is_valid:
                print(f"  ERROR: {error_msg}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            results['detailed'].append({
                'id': test_id,
                'config': test_config,
                'error': str(e),
                'valid': False
            })
        
        print()
    
    # Calculate summary statistics
    valid_results = [r for r in results['detailed'] if r.get('valid', False)]
    
    if valid_results:
        import numpy as np
        
        improvements = [r['improvement_pct'] for r in valid_results]
        speedups = [r['speedup'] for r in valid_results]
        times = [r['solution_time'] for r in valid_results]
        total_times = [r['total_time'] for r in valid_results]
        
        # Group by city size
        by_size = {}
        for r in valid_results:
            size = r['config']['num_cities']
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(r)
        
        results['summary'] = {
            'total_tests': len(test_cases_list),
            'valid_solutions': len(valid_results),
            'success_rate': (len(valid_results) / len(test_cases_list)) * 100,
            'avg_improvement': np.mean(improvements),
            'min_improvement': np.min(improvements),
            'max_improvement': np.max(improvements),
            'std_improvement': np.std(improvements),
            'avg_speedup': np.mean(speedups),
            'avg_solution_time': np.mean(times),
            'avg_total_time': np.mean(total_times),
            'max_total_time': np.max(total_times),
            'by_city_size': {}
        }
        
        # Statistics by city size
        for size in sorted(by_size.keys()):
            size_results = by_size[size]
            size_improvements = [r['improvement_pct'] for r in size_results]
            size_times = [r['solution_time'] for r in size_results]
            
            results['summary']['by_city_size'][size] = {
                'count': len(size_results),
                'avg_improvement': np.mean(size_improvements),
                'avg_time': np.mean(size_times),
                'min_time': np.min(size_times),
                'max_time': np.max(size_times)
            }
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Valid Solutions: {results['summary']['valid_solutions']} "
              f"({results['summary']['success_rate']:.1f}%)")
        print(f"Average Improvement: {results['summary']['avg_improvement']:.2f}%")
        print(f"Improvement Range: {results['summary']['min_improvement']:.2f}% to "
              f"{results['summary']['max_improvement']:.2f}%")
        print(f"Average Speedup: {results['summary']['avg_speedup']:.2f}x")
        print(f"Average Solution Time: {results['summary']['avg_solution_time']:.2f}s")
        print(f"Average Total Time: {results['summary']['avg_total_time']:.2f}s")
        print(f"Max Total Time: {results['summary']['max_total_time']:.2f}s")
        print()
        print("Performance by City Size:")
        print(f"{'Size':<8} {'Tests':<8} {'Avg Improvement':<18} {'Avg Time (s)':<15} {'Time Range (s)':<20}")
        print("-" * 80)
        for size in sorted(results['summary']['by_city_size'].keys()):
            stats = results['summary']['by_city_size'][size]
            print(f"{size:<8} {stats['count']:<8} {stats['avg_improvement']:<18.2f} "
                  f"{stats['avg_time']:<15.2f} {stats['min_time']:.2f}-{stats['max_time']:.2f}")
        print()
    
    # Save results to JSON
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    json_path = output_path / "large_scale_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {json_path}")
    
    return results

if __name__ == "__main__":
    test_cases = generate_large_scale_test_cases()
    print(f"Generated {len(test_cases)} large scale test cases")
    print("City sizes: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000")
    print("Each size tested with: α=1.0 β=1.5, α=1.0 β=2.0, α=2.0 β=2.0")
    print()
    results = run_large_scale_tests(test_cases)
PYTHON_SCRIPT

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Large scale tests completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: outputs/large_scale_results.json"
    echo ""
    echo "To view results:"
    echo "  cat outputs/large_scale_results.json | python -m json.tool"
    echo ""
    echo "To view summary statistics:"
    echo "  python -c \"import json; data=json.load(open('outputs/large_scale_results.json')); print('Success Rate:', data['summary']['success_rate'], '%'); print('Avg Improvement:', data['summary']['avg_improvement'], '%'); print('Avg Time:', data['summary']['avg_solution_time'], 's')\""
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "✗ Large scale tests failed!"
    echo "================================================================================"
    exit 1
fi
