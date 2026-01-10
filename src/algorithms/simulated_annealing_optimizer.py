import numpy as np
from typing import List, Tuple
import time
import math

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour


class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self, problem, max_iterations: int = 10000, verbose: bool = False, seed: int = None,
                 initial_temp: float = 1000.0, cooling_rate: float = 0.995, min_temp: float = 0.01):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.temperature = initial_temp
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        tour = nearest_neighbor_tour(self.problem, randomize=False, rng=self.rng)
        
        solution = []
        for city in tour:
            max_gold = self.problem._graph.nodes[city]['gold']
            gold_ratio = 0.5 + self.rng.random() * 0.3
            solution.append((city, max_gold * gold_ratio))
        
        solution.append((0, 0))
        return solution
    
    def get_neighbor(self, solution: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        neighbor = solution.copy()
        
        tour = [(city, gold) for city, gold in neighbor if city != 0]
        
        if len(tour) < 2:
            return solution
        
        move = self.rng.integers(0, 5)
        
        if move == 0:
            if len(tour) > 1:
                i, j = self.rng.choice(len(tour), 2, replace=False)
                tour[i], tour[j] = tour[j], tour[i]
        
        elif move == 1:
            idx = self.rng.integers(0, len(tour))
            city, gold = tour[idx]
            max_gold = self.problem._graph.nodes[city]['gold']
            current_ratio = gold / max_gold
            delta = self.rng.uniform(-0.15, 0.15)
            new_ratio = np.clip(current_ratio + delta, 0.0, 1.0)
            tour[idx] = (city, max_gold * new_ratio)
        
        elif move == 2:
            if len(tour) > 2:
                from_idx = self.rng.integers(0, len(tour))
                to_idx = self.rng.integers(0, len(tour))
                if from_idx != to_idx:
                    city_gold = tour.pop(from_idx)
                    tour.insert(to_idx, city_gold)
        
        elif move == 3:
            if len(tour) > 2 and tour.count((0, 0)) < 3:  # Limit depot returns
                insert_pos = self.rng.integers(1, len(tour))
                tour.insert(insert_pos, (0, 0))
        
        else:  # move == 4
            depot_indices = [i for i, (c, g) in enumerate(tour) if c == 0]
            if depot_indices:
                idx = self.rng.choice(depot_indices)
                tour.pop(idx)
        
        neighbor = tour + [(0, 0)]
        return neighbor
    
    def accept_solution(self, current_cost: float, new_cost: float) -> bool:
        if new_cost < current_cost:
            return True
        
        delta = new_cost - current_cost
        probability = math.exp(-delta / self.temperature)
        return self.rng.random() < probability
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        self.start_time = time.time()
        self.log("Starting Simulated Annealing")
        
        current_solution = self.generate_initial_solution()
        current_cost = evaluate_solution(self.problem, current_solution)
        
        self.update_best(current_solution, current_cost)
        self.temperature = self.initial_temp
        
        accepts = 0
        rejects = 0
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            new_solution = self.get_neighbor(current_solution)
            new_cost = evaluate_solution(self.problem, new_solution)
            
            if self.accept_solution(current_cost, new_cost):
                current_solution = new_solution
                current_cost = new_cost
                accepts += 1
                
                if new_cost < self.best_cost:
                    self.update_best(new_solution, new_cost)
            else:
                rejects += 1
            
            self.temperature *= self.cooling_rate
            
            if self.temperature < self.min_temp:
                self.log(f"Temperature below minimum, stopping at iteration {iteration}")
                break
            
            if iteration % 1000 == 0:
                accept_rate = accepts / (accepts + rejects) if (accepts + rejects) > 0 else 0
                self.log(f"Iter {iteration}: Best={self.best_cost:.2f}, Current={current_cost:.2f}, "
                        f"Temp={self.temperature:.2f}, Accept={accept_rate:.2%}")
                accepts = rejects = 0
        
        self.log(f"Final best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost

