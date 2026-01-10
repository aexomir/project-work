"""
Base optimizer class for TTP algorithms
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import time


class BaseOptimizer(ABC):
    """Abstract base class for TTP optimizers"""
    
    def __init__(self, problem, max_iterations: int = 1000, verbose: bool = False, seed: int = None):
        """
        Initialize optimizer.
        
        Args:
            problem: Problem instance
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.seed = seed
        self.best_solution = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.start_time = None
        self.history = []  # Track cost over iterations
    
    @abstractmethod
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        """
        Run the optimization algorithm.
        
        Returns:
            (solution, cost) where solution is [(city, gold), ..., (0, 0)]
        """
        pass
    
    @abstractmethod
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        """
        Generate an initial solution.
        
        Returns:
            Initial solution as [(city, gold), ..., (0, 0)]
        """
        pass
    
    def log(self, message: str):
        """Log message if verbose is enabled"""
        if self.verbose:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[{self.__class__.__name__}] {elapsed:.1f}s - {message}")
    
    def update_best(self, solution: List[Tuple[int, float]], cost: float):
        """Update best solution if this one is better"""
        if cost < self.best_cost:
            self.best_solution = solution.copy()
            self.best_cost = cost
            self.log(f"New best: {cost:.2f} (iteration {self.iteration})")
            return True
        return False
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since optimization started"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

