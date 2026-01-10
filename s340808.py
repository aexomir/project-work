"""
Solution file for TTP optimization
Student ID: s340808
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from Problem import Problem
from algorithms.conservative_optimizer import ConservativeOptimizer


def solution(p: Problem):
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
