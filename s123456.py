"""
Solution file for TTP optimization
Student ID: s123456 (replace with your actual student ID)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from optimizers.conservative_optimizer import ConservativeOptimizer


def solution(p):
    """
    Solve the Traveling Thief Problem using Conservative Optimizer.
    
    Args:
        p: Problem instance
    
    Returns:
        List of (city, gold) tuples representing the optimal route,
        ending with (0, 0) to return to depot.
        Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    """
    # Use Conservative Optimizer - key strategy:
    # - Collect minimal gold early (10-20%) to avoid weight penalty
    # - Collect more gold later (60-80%) when closer to depot
    # - Frequent depot returns when weight threshold exceeded
    # - 2-opt improved tour for better route
    optimizer = ConservativeOptimizer(
        p,
        max_iterations=1,
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
