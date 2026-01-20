from abc import ABC, abstractmethod
from typing import List, Tuple
import time


class BaseOptimizer(ABC):
    def __init__(self, problem, max_iterations: int = 1000, verbose: bool = False, seed: int = None):
        self.problem = problem
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.seed = seed
        self.best_solution = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.start_time = None
        self.history = []
    
    @abstractmethod
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        pass
    
    @abstractmethod
    def generate_initial_solution(self) -> List[Tuple[int, float]]:

        pass
    
    def log(self, message: str):
        if self.verbose:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[{self.__class__.__name__}] {elapsed:.1f}s - {message}")
    
    def update_best(self, solution: List[Tuple[int, float]], cost: float):
        if cost < self.best_cost:
            self.best_solution = solution.copy()
            self.best_cost = cost
            self.log(f"New best: {cost:.2f} (iteration {self.iteration})")
            return True
        return False
    
    def get_elapsed_time(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

