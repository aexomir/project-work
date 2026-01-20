import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline.baseline_impl import Problem
from algorithms.conservative_optimizer import ConservativeOptimizer
from utils.evaluator import validate_solution
from generate_test_cases import test_cases


def run_conservative_comparison(test_cases_list, output_dir="outputs"):
    results = {
        'summary': {},
        'detailed': []
    }
    
    print("=" * 80)
    print("CONSERVATIVE OPTIMIZER COMPARISON")
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
            
            problem = Problem(
                num_cities=test_config['num_cities'],
                alpha=test_config['alpha'],
                beta=test_config['beta'],
                density=test_config['density'],
                seed=test_config['seed']
            )
            
            
            baseline_start = time.time()
            baseline_cost = problem.baseline()
            baseline_time = time.time() - baseline_start
            
            
            optimizer = ConservativeOptimizer(
                problem,
                max_iterations=1,
                verbose=False,
                seed=test_config['seed']
            )
            
            solution_start = time.time()
            solution, solution_cost = optimizer.optimize()
            solution_time = time.time() - solution_start
            
            
            is_valid, error_msg = validate_solution(problem, solution)
            
            
            if baseline_cost > 0:
                improvement = ((baseline_cost - solution_cost) / baseline_cost) * 100
                speedup = baseline_cost / solution_cost if solution_cost > 0 else 0
            else:
                improvement = 0.0
                speedup = 0.0
            
            
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
                'improvement_pct': improvement,
                'speedup': speedup,
                'valid': is_valid,
                'error': error_msg if not is_valid else None
            }
            
            results['detailed'].append(test_result)
            
            
            status = "✓" if is_valid else "✗"
            print(f"  {status} Baseline: {baseline_cost:,.2f}, "
                  f"Solution: {solution_cost:,.2f}, "
                  f"Improvement: {improvement:.2f}%, "
                  f"Time: {solution_time:.4f}s")
            
            if not is_valid:
                print(f"  ERROR: {error_msg}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            results['detailed'].append({
                'id': test_id,
                'config': test_config,
                'error': str(e),
                'valid': False
            })
        
        print()
    
    
    valid_results = [r for r in results['detailed'] if r.get('valid', False)]
    
    if valid_results:
        import numpy as np
        
        improvements = [r['improvement_pct'] for r in valid_results]
        speedups = [r['speedup'] for r in valid_results]
        times = [r['solution_time'] for r in valid_results]
        
        results['summary'] = {
            'total_tests': len(test_cases_list),
            'valid_solutions': len(valid_results),
            'success_rate': (len(valid_results) / len(test_cases_list)) * 100,
            'avg_improvement': np.mean(improvements),
            'min_improvement': np.min(improvements),
            'max_improvement': np.max(improvements),
            'std_improvement': np.std(improvements),
            'avg_speedup': np.mean(speedups),
            'min_speedup': np.min(speedups),
            'max_speedup': np.max(speedups),
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
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
        print(f"Average Time: {results['summary']['avg_time']:.4f}s")
        print()
    
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    json_path = output_path / "conservative_comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {json_path}")
    
    return results


def main():
    """Main entry point"""
    results = run_conservative_comparison(test_cases)
    return results


if __name__ == "__main__":
    main()
