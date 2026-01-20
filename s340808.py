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
    optimizer = ConservativeOptimizer(
        p,
        max_iterations=1,
        verbose=False,
        seed=42
    )
    
    route, cost = optimizer.optimize()
    return route
