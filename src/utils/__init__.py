"""
Utility functions for TTP optimization and algorithm comparison

This package provides utility functions used by the algorithm comparison
framework for evaluating solutions, validating results, and extracting
performance statistics.
"""

from .evaluator import evaluate_solution, validate_solution, calculate_path_cost
from .helpers import nearest_neighbor_tour, two_opt_improve, extract_route_stats

__all__ = [
    'evaluate_solution',
    'validate_solution',
    'calculate_path_cost',
    'nearest_neighbor_tour',
    'two_opt_improve',
    'extract_route_stats'
]

