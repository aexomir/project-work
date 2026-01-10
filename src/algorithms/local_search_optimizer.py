"""
Local Search / Hill Climbing optimizer for TTP
"""

import numpy as np
from typing import List, Tuple
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour, two_opt_improve


class LocalSearchOptimizer(BaseOptimizer):
    """Multi-start local search for TTP"""
    
    def __init__(self, problem, max_iterations: int = 50, verbose: bool = False, seed: int = None,
                 max_local_iterations: int = 200):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
        self.max_local_iterations = max_local_iterations
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        """Generate randomized initial solution"""
        tour = nearest_neighbor_tour(self.problem, randomize=True, rng=self.rng)
        
        solution = []
        for city in tour:
            max_gold = self.problem._graph.nodes[city]['gold']
            gold_ratio = self.rng.random()
            solution.append((city, max_gold * gold_ratio))
        
        solution.append((0, 0))
        return solution
    
    def local_search_tour(self, solution: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Apply 2-opt to improve tour order"""
        # Extract tour and gold amounts
        tour_data = [(city, gold) for city, gold in solution if city != 0]
        tour = [city for city, _ in tour_data]
        gold_dict = {city: gold for city, gold in tour_data}
        
        # Apply 2-opt
        improved_tour = two_opt_improve(self.problem, tour, max_iterations=20)
        
        # Rebuild solution with improved tour
        new_solution = []
        for city in improved_tour:
            new_solution.append((city, gold_dict[city]))
        new_solution.append((0, 0))
        
        return new_solution
    
    def local_search_gold(self, solution: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Adjust gold collection amounts"""
        improved = solution.copy()
        current_cost = evaluate_solution(self.problem, improved)
        
        # Extract tour without depot
        tour_data = [(city, gold) for city, gold in improved if city != 0]
        
        # Try adjusting each city's gold
        for i in range(len(tour_data)):
            city, current_gold = tour_data[i]
            max_gold = self.problem._graph.nodes[city]['gold']
            current_ratio = current_gold / max_gold
            
            # Try different gold ratios around current value
            for delta in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
                new_ratio = np.clip(current_ratio + delta, 0.0, 1.0)
                new_gold = max_gold * new_ratio
                
                # Create test solution
                test_solution = []
                for j, (c, g) in enumerate(tour_data):
                    if j == i:
                        test_solution.append((c, new_gold))
                    else:
                        test_solution.append((c, g))
                test_solution.append((0, 0))
                
                # Evaluate
                test_cost = evaluate_solution(self.problem, test_solution)
                if test_cost < current_cost:
                    improved = test_solution
                    current_cost = test_cost
                    tour_data[i] = (city, new_gold)
                    break
        
        return improved
    
    def local_search_depot_returns(self, solution: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Try adding/removing depot returns"""
        improved = solution.copy()
        current_cost = evaluate_solution(self.problem, improved)
        
        # Extract tour without final depot
        tour = [item for item in improved[:-1]]
        
        # Try adding depot return at various positions
        for pos in range(1, len(tour)):
            test_solution = tour[:pos] + [(0, 0)] + tour[pos:] + [(0, 0)]
            test_cost = evaluate_solution(self.problem, test_solution)
            if test_cost < current_cost:
                improved = test_solution
                current_cost = test_cost
        
        # Try removing existing depot returns
        depot_positions = [i for i, (c, g) in enumerate(tour) if c == 0]
        for pos in depot_positions:
            test_solution = tour[:pos] + tour[pos+1:] + [(0, 0)]
            test_cost = evaluate_solution(self.problem, test_solution)
            if test_cost < current_cost:
                improved = test_solution
                current_cost = test_cost
        
        return improved
    
    def local_search(self, initial_solution: List[Tuple[int, float]]) -> Tuple[List[Tuple[int, float]], float]:
        """Perform local search from initial solution"""
        current = initial_solution
        current_cost = evaluate_solution(self.problem, current)
        
        for local_iter in range(self.max_local_iterations):
            improved = False
            
            # Try tour improvement
            new_solution = self.local_search_tour(current)
            new_cost = evaluate_solution(self.problem, new_solution)
            if new_cost < current_cost:
                current = new_solution
                current_cost = new_cost
                improved = True
            
            # Try gold adjustment
            new_solution = self.local_search_gold(current)
            new_cost = evaluate_solution(self.problem, new_solution)
            if new_cost < current_cost:
                current = new_solution
                current_cost = new_cost
                improved = True
            
            # Try depot return adjustment
            new_solution = self.local_search_depot_returns(current)
            new_cost = evaluate_solution(self.problem, new_solution)
            if new_cost < current_cost:
                current = new_solution
                current_cost = new_cost
                improved = True
            
            # Stop if no improvement
            if not improved:
                break
        
        return current, current_cost
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        """Run multi-start local search"""
        self.start_time = time.time()
        self.log("Starting Multi-Start Local Search")
        
        # Try multiple starting points
        for start in range(self.max_iterations):
            self.iteration = start
            
            # Generate initial solution
            initial = self.generate_initial_solution()
            
            # Perform local search
            solution, cost = self.local_search(initial)
            
            # Update best
            if cost < self.best_cost:
                self.update_best(solution, cost)
            
            # Log progress
            if start % 10 == 0:
                self.log(f"Start {start}/{self.max_iterations}: Best={self.best_cost:.2f}, Current={cost:.2f}")
        
        self.log(f"Final best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost

