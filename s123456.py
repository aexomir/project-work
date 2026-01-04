"""
Solution file for TTP optimization
Student ID: s123456 (replace with your actual student ID)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from optimizers.genetic_optimizer import GeneticOptimizer


def solution(p):
    """
    Solve the Traveling Thief Problem using Genetic Algorithm.
    
    Args:
        p: Problem instance
    
    Returns:
        List of (city, gold) tuples representing the optimal route,
        ending with (0, 0) to return to depot.
        Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    """
    # Use Genetic Algorithm as it typically provides best balance of quality and runtime
    # Parameters tuned for good performance across different problem sizes
    optimizer = GeneticOptimizer(
        p,
        max_iterations=500,
        population_size=150,
        mutation_rate=0.15,
        crossover_rate=0.8,
        verbose=False,
        seed=42
    )
    
    route, cost = optimizer.optimize()
    
    return route


# Alternative: Use Simulated Annealing for potentially better solutions (slower)
def solution_sa(p):
    """Alternative solution using Simulated Annealing"""
    from optimizers.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
    
    optimizer = SimulatedAnnealingOptimizer(
        p,
        max_iterations=10000,
        initial_temp=1000.0,
        cooling_rate=0.995,
        verbose=False,
        seed=42
    )
    
    route, cost = optimizer.optimize()
    return route


# Alternative: Use Local Search for best quality
def solution_ls(p):
    """Alternative solution using Local Search"""
    from optimizers.local_search_optimizer import LocalSearchOptimizer
    
    optimizer = LocalSearchOptimizer(
        p,
        max_iterations=50,
        verbose=False,
        seed=42
    )
    
    route, cost = optimizer.optimize()
    return route
