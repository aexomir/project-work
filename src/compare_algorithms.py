"""
Algorithm Comparison Framework for TTP Optimization

This module provides a comprehensive framework for comparing different optimization
algorithms on the Traveling Thief Problem (TTP). It runs multiple algorithms on
various problem configurations and generates detailed comparison reports.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline.baseline_impl import Problem
from algorithms.greedy_optimizer import GreedyOptimizer
from algorithms.conservative_optimizer import ConservativeOptimizer
from algorithms.genetic_optimizer import GeneticOptimizer
from algorithms.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from algorithms.local_search_optimizer import LocalSearchOptimizer
from algorithms.aco_optimizer import ACOOptimizer
from utils.evaluator import validate_solution
from utils.helpers import extract_route_stats


def compare_algorithms(problem_configs, output_dir="outputs"):
    """
    Compare multiple optimization algorithms on test problem configurations.
    
    This function runs each algorithm in the comparison suite on all provided
    problem configurations, collects performance metrics, and generates a
    comprehensive comparison report.
    
    Args:
        problem_configs: List of problem configuration dictionaries to test
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary containing:
            - 'summary': Aggregated comparison statistics per algorithm
            - 'detailed': Per-test-case comparison results for each algorithm
    """
    comparison_results = {
        'summary': {},
        'detailed': []
    }
    
    # Algorithm comparison suite (ordered from fastest to slowest)
    algorithms_to_compare = [
        ('Greedy', GreedyOptimizer, {'max_iterations': 1, 'verbose': True}),
        ('Conservative', ConservativeOptimizer, {'max_iterations': 1, 'verbose': True}),
        # ('SimulatedAnnealing', SimulatedAnnealingOptimizer, {'max_iterations': 5000, 'verbose': True}),
        ('GeneticAlgorithm', GeneticOptimizer, {'max_iterations': 300, 'verbose': True, 'population_size': 100}),
        # ('AntColony', ACOOptimizer, {'max_iterations': 150, 'verbose': True, 'num_ants': 40}),
        # ('LocalSearch', LocalSearchOptimizer, {'max_iterations': 20, 'verbose': True}),
    ]
    
    print("=" * 80)
    print("TTP OPTIMIZATION ALGORITHM COMPARISON")
    print("=" * 80)
    print()
    
    for config_idx, config in enumerate(problem_configs):
        print(f"\n{'='*80}")
        print(f"TEST CASE {config_idx + 1}: {config}")
        print(f"{'='*80}\n")
        
        # Create problem instance
        problem = Problem(**config)
        
        # Calculate baseline cost for comparison
        baseline_cost = problem.baseline()
        print(f"Baseline cost: {baseline_cost:.2f}\n")
        
        # Store comparison results for this test case
        test_case_comparison = {
            'config': config,
            'baseline_cost': baseline_cost,
            'algorithms': {}
        }
        
        # Run each algorithm in the comparison suite
        for algo_name, algo_class, algo_params in algorithms_to_compare:
            print(f"\n{'-'*80}")
            print(f"Running {algo_name}...")
            print(f"{'-'*80}")
            
            try:
                # Create optimizer
                optimizer = algo_class(problem, seed=42, **algo_params)
                
                # Run optimization
                start_time = time.time()
                solution, cost = optimizer.optimize()
                elapsed_time = time.time() - start_time
                
                # Validate solution
                is_valid, error_msg = validate_solution(problem, solution)
                
                # Extract statistics
                stats = extract_route_stats(problem, solution) if is_valid else {}
                
                # Calculate improvement over baseline for comparison
                improvement = ((baseline_cost - cost) / baseline_cost * 100) if baseline_cost > 0 else 0
                
                # Store algorithm comparison metrics
                algorithm_comparison_result = {
                    'cost': cost,
                    'time': elapsed_time,
                    'valid': is_valid,
                    'error': error_msg if not is_valid else None,
                    'improvement_pct': improvement,
                    'solution_length': len(solution),
                    'stats': stats
                }
                
                test_case_comparison['algorithms'][algo_name] = algorithm_comparison_result
                
                # Print summary
                print(f"\n✓ {algo_name} completed in {elapsed_time:.2f}s")
                print(f"  Cost: {cost:.2f}")
                print(f"  Improvement: {improvement:.2f}%")
                print(f"  Valid: {is_valid}")
                if is_valid and stats:
                    print(f"  Depot returns: {stats['depot_returns']}")
                    print(f"  Avg gold ratio: {stats['avg_gold_ratio']:.2%}")
                
            except Exception as e:
                print(f"\n✗ {algo_name} failed: {str(e)}")
                test_case_comparison['algorithms'][algo_name] = {
                    'error': str(e),
                    'cost': float('inf'),
                    'valid': False
                }
        
        comparison_results['detailed'].append(test_case_comparison)
        
        # Print algorithm comparison table for this test case
        print(f"\n{'-'*80}")
        print(f"ALGORITHM COMPARISON FOR TEST CASE {config_idx + 1}")
        print(f"{'-'*80}")
        print(f"{'Algorithm':<20} {'Cost':>15} {'Time (s)':>10} {'Improvement':>12} {'Valid':>8}")
        print(f"{'-'*80}")
        print(f"{'Baseline':<20} {baseline_cost:>15.2f} {'-':>10} {'-':>12} {'N/A':>8}")
        
        for algo_name in [name for name, _, _ in algorithms_to_compare]:
            if algo_name in test_case_comparison['algorithms']:
                res = test_case_comparison['algorithms'][algo_name]
                if 'cost' in res and res['cost'] != float('inf'):
                    valid_str = '✓' if res['valid'] else '✗'
                    print(f"{algo_name:<20} {res['cost']:>15.2f} {res['time']:>10.2f} "
                          f"{res['improvement_pct']:>11.2f}% {valid_str:>8}")
                else:
                    print(f"{algo_name:<20} {'ERROR':>15} {'-':>10} {'-':>12} {'✗':>8}")
    
    # Calculate overall algorithm comparison summary
    print(f"\n{'='*80}")
    print("OVERALL ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    for algo_name in [name for name, _, _ in algorithms_to_compare]:
        valid_costs = []
        valid_times = []
        valid_improvements = []
        
        for test_case_result in comparison_results['detailed']:
            if algo_name in test_case_result['algorithms']:
                algo_result = test_case_result['algorithms'][algo_name]
                if algo_result['valid'] and algo_result['cost'] != float('inf'):
                    valid_costs.append(algo_result['cost'])
                    valid_times.append(algo_result['time'])
                    valid_improvements.append(algo_result['improvement_pct'])
        
        if valid_costs:
            import numpy as np
            comparison_results['summary'][algo_name] = {
                'avg_improvement': np.mean(valid_improvements),
                'min_improvement': np.min(valid_improvements),
                'max_improvement': np.max(valid_improvements),
                'avg_time': np.mean(valid_times),
                'success_rate': len(valid_costs) / len(comparison_results['detailed']) * 100
            }
            
            print(f"{algo_name}:")
            print(f"  Avg improvement: {comparison_results['summary'][algo_name]['avg_improvement']:.2f}%")
            print(f"  Range: {comparison_results['summary'][algo_name]['min_improvement']:.2f}% to "
                  f"{comparison_results['summary'][algo_name]['max_improvement']:.2f}%")
            print(f"  Avg time: {comparison_results['summary'][algo_name]['avg_time']:.2f}s")
            print(f"  Success rate: {comparison_results['summary'][algo_name]['success_rate']:.1f}%")
            print()
    
    # Determine best performing algorithm from comparison
    best_algorithm = max(comparison_results['summary'].items(), 
                        key=lambda x: x[1]['avg_improvement'])
    print(f"{'='*80}")
    print(f"BEST ALGORITHM (from comparison): {best_algorithm[0]} "
          f"(avg improvement: {best_algorithm[1]['avg_improvement']:.2f}%)")
    print(f"{'='*80}\n")
    
    # Save comparison results to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save algorithm comparison results as JSON
    json_path = output_path / "alg_comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Algorithm comparison results saved to {json_path}")
    
    return comparison_results


def main():
    """
    Main entry point for algorithm comparison framework.
    
    Runs algorithm comparison on standard test configurations, with optional
    large-scale problem testing. Generates comprehensive comparison reports.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compare TTP optimization algorithms - Run comprehensive algorithm comparison'
    )
    parser.add_argument('--large', action='store_true',
                       help='Run algorithm comparison on large problems (1000 cities) in addition to standard tests')
    args = parser.parse_args()
    
    # Standard test configurations for algorithm comparison
    # Density from 0.5 to 2.0 in 0.5 increments
    standard_test_configs = []
    base_num_cities = 100
    for density in [0.5, 1.0, 1.5, 2.0]:
        for alpha in [1, 2]:
            for beta in [1, 2]:
                standard_test_configs.append({
                    'num_cities': base_num_cities,
                    'density': density,
                    'alpha': alpha,
                    'beta': beta,
                    'seed': 42
                })
    
    print(f"Running algorithm comparison on {base_num_cities}-city problems...")
    standard_comparison_results = compare_algorithms(standard_test_configs)
    
    # Run algorithm comparison on large test configurations if requested
    if args.large:
        print("\n\nRunning algorithm comparison on 1000-city problems...")
        large_test_configs = []
        for density in [0.5, 1.0, 1.5, 2.0]:
            for alpha in [1, 2]:
                for beta in [1, 2]:
                    large_test_configs.append({
                        'num_cities': 1000,
                        'density': density,
                        'alpha': alpha,
                        'beta': beta,
                        'seed': 42
                    })
        large_comparison_results = compare_algorithms(large_test_configs)
        return {
            'standard': standard_comparison_results,
            'large': large_comparison_results
        }
    
    return standard_comparison_results


if __name__ == "__main__":
    main()

