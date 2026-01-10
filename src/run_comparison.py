"""
Comparison framework for TTP optimization algorithms
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline.baseline_impl import Problem
from optimizers.greedy_optimizer import GreedyOptimizer
from optimizers.conservative_optimizer import ConservativeOptimizer
from optimizers.genetic_optimizer import GeneticOptimizer
from optimizers.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from optimizers.local_search_optimizer import LocalSearchOptimizer
from optimizers.aco_optimizer import ACOOptimizer
from utils.evaluator import validate_solution
from utils.helpers import extract_route_stats


def compare_algorithms(problem_configs, output_dir="outputs"):
    """
    Run all optimizers on test cases and compare results.
    
    Args:
        problem_configs: List of problem configuration dicts
        output_dir: Directory to save results
    
    Returns:
        Dictionary with all results
    """
    results = {
        'summary': {},
        'detailed': []
    }
    
    # Algorithm configurations (ordered from fastest to slowest)
    algorithms = [
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
        
        # Calculate baseline cost
        baseline_cost = problem.baseline()
        print(f"Baseline cost: {baseline_cost:.2f}\n")
        
        config_results = {
            'config': config,
            'baseline_cost': baseline_cost,
            'algorithms': {}
        }
        
        # Run each algorithm
        for algo_name, algo_class, algo_params in algorithms:
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
                
                # Calculate improvement
                improvement = ((baseline_cost - cost) / baseline_cost * 100) if baseline_cost > 0 else 0
                
                # Store results
                algo_results = {
                    'cost': cost,
                    'time': elapsed_time,
                    'valid': is_valid,
                    'error': error_msg if not is_valid else None,
                    'improvement_pct': improvement,
                    'solution_length': len(solution),
                    'stats': stats
                }
                
                config_results['algorithms'][algo_name] = algo_results
                
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
                config_results['algorithms'][algo_name] = {
                    'error': str(e),
                    'cost': float('inf'),
                    'valid': False
                }
        
        results['detailed'].append(config_results)
        
        # Print comparison table for this config
        print(f"\n{'-'*80}")
        print(f"COMPARISON FOR TEST CASE {config_idx + 1}")
        print(f"{'-'*80}")
        print(f"{'Algorithm':<20} {'Cost':>15} {'Time (s)':>10} {'Improvement':>12} {'Valid':>8}")
        print(f"{'-'*80}")
        print(f"{'Baseline':<20} {baseline_cost:>15.2f} {'-':>10} {'-':>12} {'N/A':>8}")
        
        for algo_name in [name for name, _, _ in algorithms]:
            if algo_name in config_results['algorithms']:
                res = config_results['algorithms'][algo_name]
                if 'cost' in res and res['cost'] != float('inf'):
                    valid_str = '✓' if res['valid'] else '✗'
                    print(f"{algo_name:<20} {res['cost']:>15.2f} {res['time']:>10.2f} "
                          f"{res['improvement_pct']:>11.2f}% {valid_str:>8}")
                else:
                    print(f"{algo_name:<20} {'ERROR':>15} {'-':>10} {'-':>12} {'✗':>8}")
    
    # Calculate overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    for algo_name in [name for name, _, _ in algorithms]:
        valid_costs = []
        valid_times = []
        valid_improvements = []
        
        for config_result in results['detailed']:
            if algo_name in config_result['algorithms']:
                res = config_result['algorithms'][algo_name]
                if res['valid'] and res['cost'] != float('inf'):
                    valid_costs.append(res['cost'])
                    valid_times.append(res['time'])
                    valid_improvements.append(res['improvement_pct'])
        
        if valid_costs:
            import numpy as np
            results['summary'][algo_name] = {
                'avg_improvement': np.mean(valid_improvements),
                'min_improvement': np.min(valid_improvements),
                'max_improvement': np.max(valid_improvements),
                'avg_time': np.mean(valid_times),
                'success_rate': len(valid_costs) / len(results['detailed']) * 100
            }
            
            print(f"{algo_name}:")
            print(f"  Avg improvement: {results['summary'][algo_name]['avg_improvement']:.2f}%")
            print(f"  Range: {results['summary'][algo_name]['min_improvement']:.2f}% to "
                  f"{results['summary'][algo_name]['max_improvement']:.2f}%")
            print(f"  Avg time: {results['summary'][algo_name]['avg_time']:.2f}s")
            print(f"  Success rate: {results['summary'][algo_name]['success_rate']:.1f}%")
            print()
    
    # Find best algorithm
    best_algo = max(results['summary'].items(), 
                   key=lambda x: x[1]['avg_improvement'])
    print(f"{'='*80}")
    print(f"BEST ALGORITHM: {best_algo[0]} (avg improvement: {best_algo[1]['avg_improvement']:.2f}%)")
    print(f"{'='*80}\n")
    
    # Save results to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save JSON results
    json_path = output_path / "alg_comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    return results


def main():
    """Run comparison on standard test configurations"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare TTP optimization algorithms')
    parser.add_argument('--large', action='store_true',
                       help='Run comparison on large problems (1000 cities) in addition to standard tests')
    args = parser.parse_args()
    
    # Standard test configurations with density from 0.5 to 2.0 in 0.5 increments
    test_configs = []
    base_num_cities = 100
    for density in [0.5, 1.0, 1.5, 2.0]:
        for alpha in [1, 2]:
            for beta in [1, 2]:
                test_configs.append({
                    'num_cities': base_num_cities,
                    'density': density,
                    'alpha': alpha,
                    'beta': beta,
                    'seed': 42
                })
    
    print(f"Running comparison on {base_num_cities}-city problems...")
    results_base = compare_algorithms(test_configs)
    
    # Run large test configurations only if --large flag is passed
    if args.large:
        print("\n\nRunning comparison on 1000-city problems...")
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
        results_large = compare_algorithms(large_test_configs)
        return {'standard': results_base, 'large': results_large}
    
    return results_base


if __name__ == "__main__":
    main()

