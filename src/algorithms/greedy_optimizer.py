"""
Greedy optimizer for TTP
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour


class GreedyOptimizer(BaseOptimizer):
    """Greedy heuristic optimizer for TTP"""
    
    def __init__(self, problem, max_iterations: int = 1, verbose: bool = False, seed: int = None):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        """
        Generate greedy solution:
        1. Build tour using nearest neighbor
        2. Collect gold based on distance-to-depot heuristic
        3. Insert depot returns when weight is high
        """
        # Get nearest neighbor tour
        tour = nearest_neighbor_tour(self.problem, start_city=0, randomize=False)
        
        # Build solution with greedy gold collection
        solution = []
        
        # Calculate distance from each city to depot
        city_to_depot_dist = {}
        for city in tour:
            if self.problem._graph.has_edge(city, 0):
                city_to_depot_dist[city] = self.problem._graph.edges[city, 0]['dist']
            else:
                try:
                    path = nx.shortest_path(self.problem._graph, city, 0, weight='dist')
                    city_to_depot_dist[city] = nx.path_weight(self.problem._graph, path, weight='dist')
                except nx.NetworkXNoPath:
                    city_to_depot_dist[city] = float('inf')
        
        # Calculate total remaining distance for each position in tour
        total_dist = 0.0
        for i in range(len(tour)):
            c1 = tour[i-1] if i > 0 else 0
            c2 = tour[i]
            if self.problem._graph.has_edge(c1, c2):
                total_dist += self.problem._graph.edges[c1, c2]['dist']
        
        # Strategy: collect less gold if we have far to go
        current_weight = 0.0
        weight_threshold = 50.0 * self.problem.alpha  # Adaptive threshold based on alpha
        
        for i, city in enumerate(tour):
            # Calculate remaining tour length
            remaining_ratio = (len(tour) - i) / len(tour)
            
            # Get max gold at this city
            max_gold = self.problem._graph.nodes[city]['gold']
            
            # Greedy strategy: collect more gold if close to depot or early in tour
            # Collect less if carrying a lot already
            if current_weight > weight_threshold and remaining_ratio > 0.3:
                # Heavy load and far from end - return to depot
                solution.append((0, 0))
                current_weight = 0.0
            
            # Gold collection ratio based on:
            # 1. Distance to depot (closer = collect more)
            # 2. Position in tour (later = collect more)
            # 3. Current weight (heavier = collect less)
            
            dist_factor = 1.0 / (1.0 + city_to_depot_dist[city] / 10.0)
            position_factor = 1.0 - remaining_ratio * 0.5
            weight_factor = max(0.3, 1.0 - current_weight / (weight_threshold * 2))
            
            gold_ratio = dist_factor * position_factor * weight_factor
            gold_ratio = np.clip(gold_ratio, 0.1, 1.0)
            
            gold_to_collect = max_gold * gold_ratio
            solution.append((city, gold_to_collect))
            current_weight += gold_to_collect
        
        # Return to depot
        solution.append((0, 0))
        
        return solution
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        """Run greedy optimization"""
        self.start_time = time.time()
        self.log("Starting greedy optimization")
        
        # Generate initial solution
        solution = self.generate_initial_solution()
        cost = evaluate_solution(self.problem, solution)
        
        self.update_best(solution, cost)
        self.log(f"Greedy solution cost: {cost:.2f}")
        
        return self.best_solution, self.best_cost

