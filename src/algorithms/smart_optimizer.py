import numpy as np
import networkx as nx
from typing import List, Tuple
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour, two_opt_improve


class SmartOptimizer(BaseOptimizer):
    
    def __init__(self, problem, max_iterations: int = 1, verbose: bool = False, seed: int = None):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
    
    def calculate_optimal_gold_collection(self, tour: List[int]) -> List[Tuple[int, float]]:
        num_cities = len(tour)
        if num_cities == 0:
            return [(0, 0)]
        
        distances = {}
        for i in range(num_cities):
            c1 = tour[i]
            c2 = tour[i + 1] if i < num_cities - 1 else 0
            if self.problem._graph.has_edge(c1, c2):
                distances[(c1, c2)] = self.problem._graph.edges[c1, c2]['dist']
            else:
                try:
                    path = nx.shortest_path(self.problem._graph, c1, c2, weight='dist')
                    distances[(c1, c2)] = nx.path_weight(self.problem._graph, path, weight='dist')
                except nx.NetworkXNoPath:
                    distances[(c1, c2)] = float('inf')
        
        dist_to_depot = {}
        for city in tour:
            if self.problem._graph.has_edge(city, 0):
                dist_to_depot[city] = self.problem._graph.edges[city, 0]['dist']
            else:
                try:
                    path = nx.shortest_path(self.problem._graph, city, 0, weight='dist')
                    dist_to_depot[city] = nx.path_weight(self.problem._graph, path, weight='dist')
                except nx.NetworkXNoPath:
                    dist_to_depot[city] = float('inf')
        
        solution = []
        current_weight = 0.0
        
        alpha_factor = 1.0 / (1.0 + self.problem.alpha * 0.5)
        beta_factor = 1.0 / (1.0 + self.problem.beta * 0.5)
        weight_penalty_factor = alpha_factor * beta_factor
        
        remaining_dist = {}
        total = 0.0
        for i in range(len(tour) - 1, -1, -1):
            if i < len(tour) - 1:
                c1, c2 = tour[i], tour[i + 1]
                total += distances.get((c1, c2), float('inf'))
            remaining_dist[tour[i]] = total
        
        for i, city in enumerate(tour):
            max_gold = self.problem._graph.nodes[city]['gold']
            remaining = remaining_dist[city]
            depot_dist = dist_to_depot[city]
            
            base_ratio = 0.5  # Start with 50%
            
            if depot_dist < float('inf'):
                depot_factor = 1.0 / (1.0 + depot_dist / 5.0)
            else:
                depot_factor = 0.3
            
            position_factor = 0.3 + 0.7 * (i / len(tour))
            
            if remaining > 0:
                remaining_factor = 1.0 / (1.0 + remaining / 10.0)
            else:
                remaining_factor = 1.0
            
            weight_factor = max(0.2, 1.0 - current_weight / (50.0 * weight_penalty_factor))
            
            gold_value_factor = min(1.0, max_gold / 500.0)
            
            gold_ratio = (depot_factor * 0.25 + 
                         position_factor * 0.25 + 
                         remaining_factor * 0.2 + 
                         weight_factor * 0.2 +
                         gold_value_factor * 0.1)
            
            if self.problem.alpha > 1.5 or self.problem.beta > 1.5:
                gold_ratio *= 0.5
            elif self.problem.alpha > 1.0 or self.problem.beta > 1.0:
                gold_ratio *= 0.7
            
            gold_ratio = np.clip(gold_ratio, 0.1, 0.95)
            
            best_ratio = gold_ratio
            
            gold_to_collect = max_gold * best_ratio
            
            if (current_weight > 40.0 * weight_penalty_factor and 
                remaining > 25.0 and 
                i < len(tour) - 2 and
                depot_dist < 10.0):  # Only if depot is reasonably close
                solution.append((0, 0))
                current_weight = 0.0
            
            solution.append((city, gold_to_collect))
            current_weight += gold_to_collect
        
        solution.append((0, 0))
        return solution
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        tour = nearest_neighbor_tour(self.problem, start_city=0, randomize=False)
        
        improved_tour = two_opt_improve(self.problem, tour, max_iterations=50)
        
        
        solution = self.calculate_optimal_gold_collection(improved_tour)
        
        return solution
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        self.start_time = time.time()
        self.log("Starting Smart optimization")
        
        solution = self.generate_initial_solution()
        cost = evaluate_solution(self.problem, solution)
        
        self.update_best(solution, cost)
        self.log(f"Smart solution cost: {cost:.2f}")
        
        return self.best_solution, self.best_cost

