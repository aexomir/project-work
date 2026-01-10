"""Quick test of optimizers on small problem"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline.baseline_impl import Problem
from optimizers.greedy_optimizer import GreedyOptimizer
from optimizers.genetic_optimizer import GeneticOptimizer
from optimizers.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from optimizers.local_search_optimizer import LocalSearchOptimizer
from optimizers.aco_optimizer import ACOOptimizer
from utils.evaluator import validate_solution
import time

# Small test problem
p = Problem(20, density=0.5, alpha=1, beta=1, seed=42)
baseline = p.baseline()

print(f"Baseline: {baseline:.2f}\n")

algorithms = [
    ("Greedy", GreedyOptimizer, {}),
    ("LocalSearch", LocalSearchOptimizer, {"max_iterations": 10}),
    ("SimulatedAnnealing", SimulatedAnnealingOptimizer, {"max_iterations": 1000}),
    ("GeneticAlgorithm", GeneticOptimizer, {"max_iterations": 50, "population_size": 30}),
    ("AntColony", ACOOptimizer, {"max_iterations": 30, "num_ants": 20}),
]

results = {}

for name, cls, params in algorithms:
    print(f"Testing {name}...")
    opt = cls(p, seed=42, verbose=False, **params)
    start = time.time()
    sol, cost = opt.optimize()
    elapsed = time.time() - start
    valid, msg = validate_solution(p, sol)
    improvement = (baseline - cost) / baseline * 100
    
    results[name] = {
        "cost": cost,
        "time": elapsed,
        "valid": valid,
        "improvement": improvement
    }
    
    print(f"  Cost: {cost:.2f}, Time: {elapsed:.2f}s, Valid: {valid}, Improvement: {improvement:.2f}%\n")

print("\nSummary:")
print(f"{'Algorithm':<20} {'Cost':>12} {'Time':>8} {'Improvement':>12}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<20} {res['cost']:>12.2f} {res['time']:>8.2f}s {res['improvement']:>11.2f}%")

