import numpy as np
import networkx as nx
from typing import List, Tuple
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour


class ImprovedGreedyOptimizer(BaseOptimizer):
    def __init__(self, problem, max_iterations: int = 1, verbose: bool = False, seed: int = None):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:

        tour = nearest_neighbor_tour(self.problem, start_city=0, randomize=False)
        
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
        
        remaining_dist = {}
        total_remaining = 0.0
        for i in range(len(tour) - 1, -1, -1):
            city = tour[i]
            if i < len(tour) - 1:
                next_city = tour[i + 1]
                if self.problem._graph.has_edge(city, next_city):
                    total_remaining += self.problem._graph.edges[city, next_city]['dist']
                else:
                    try:
                        path = nx.shortest_path(self.problem._graph, city, next_city, weight='dist')
                        total_remaining += nx.path_weight(self.problem._graph, path, weight='dist')
                    except nx.NetworkXNoPath:
                        pass
            remaining_dist[city] = total_remaining
        
        solution = []
        current_weight = 0.0
        
        base_threshold = 20.0
        weight_threshold = base_threshold / (1.0 + self.problem.alpha * self.problem.beta)
        
        for i, city in enumerate(tour):
            max_gold = self.problem._graph.nodes[city]['gold']
            dist_to_depot = city_to_depot_dist[city]
            remaining = remaining_dist[city]
            
            if dist_to_depot < float('inf'):
                depot_factor = 1.0 / (1.0 + dist_to_depot / 5.0)
            else:
                depot_factor = 0.3
            
            position_factor = 0.3 + 0.7 * (i / len(tour))
            
            if remaining > 0:
                remaining_factor = 1.0 / (1.0 + remaining / 10.0)
            else:
                remaining_factor = 1.0
            
            weight_factor = max(0.2, 1.0 - current_weight / (weight_threshold * 2))
            
            gold_value_factor = min(1.0, max_gold / 500.0)
            
            gold_ratio = (depot_factor * 0.3 + 
                         position_factor * 0.3 + 
                         remaining_factor * 0.2 + 
                         weight_factor * 0.15 +
                         gold_value_factor * 0.05)
            
            if self.problem.alpha > 1.5 or self.problem.beta > 1.5:
                gold_ratio *= 0.6
            elif self.problem.alpha > 1.0 or self.problem.beta > 1.0:
                gold_ratio *= 0.75
            
            gold_ratio = np.clip(gold_ratio, 0.05, 0.95)
            
            gold_to_collect = max_gold * gold_ratio
            
            if (current_weight > weight_threshold and 
                remaining > 20.0 and 
                dist_to_depot > 5.0 and
                i < len(tour) - 3):
                solution.append((0, 0))
                current_weight = 0.0
            
            solution.append((city, gold_to_collect))
            current_weight += gold_to_collect
        
        solution.append((0, 0))
        
        return solution
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        self.start_time = time.time()
        self.log("Starting Improved Greedy optimization")
        
        solution = self.generate_initial_solution()
        cost = evaluate_solution(self.problem, solution)
        
        self.update_best(solution, cost)
        self.log(f"Improved Greedy solution cost: {cost:.2f}")
        
        return self.best_solution, self.best_cost

