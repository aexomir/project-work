"""
Conservative Optimizer: Collects minimal gold early, more later
Key insight: Weight penalty is severe, so minimize early collection
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour, two_opt_improve


class ConservativeOptimizer(BaseOptimizer):
    """
    Conservative strategy:
    1. Collect very little gold early (10-20%)
    2. Collect more gold later (60-80%)
    3. Return to depot frequently when weight > threshold
    4. Use 2-opt improved tour
    """
    
    def __init__(self, problem, max_iterations: int = 1, verbose: bool = False, seed: int = None):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        """Generate conservative solution"""
        # Build and improve tour
        tour = nearest_neighbor_tour(self.problem, start_city=0, randomize=False)
        improved_tour = two_opt_improve(self.problem, tour, max_iterations=30)
        
        # Calculate distances
        dist_to_depot = {}
        for city in improved_tour:
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
        
        # Conservative weight threshold - return to depot frequently
        # Lower threshold when alpha/beta are high
        base_threshold = 15.0
        weight_threshold = base_threshold / (1.0 + (self.problem.alpha - 1.0) * 0.5 + (self.problem.beta - 1.0) * 0.5)
        
        for i, city in enumerate(improved_tour):
            max_gold = self.problem._graph.nodes[city]['gold']
            depot_dist = dist_to_depot[city]
            
            # Position in tour (0.0 = start, 1.0 = end)
            position = i / len(improved_tour)
            
            # Conservative strategy:
            # - Early cities (first 30%): collect 10-20% of gold
            # - Middle cities (30-70%): collect 30-50% of gold
            # - Late cities (last 30%): collect 60-80% of gold
            
            if position < 0.3:
                # Early: collect very little
                gold_ratio = 0.1 + 0.1 * position / 0.3  # 10-20%
            elif position < 0.7:
                # Middle: collect moderate amount
                gold_ratio = 0.2 + 0.3 * (position - 0.3) / 0.4  # 20-50%
            else:
                # Late: collect more
                gold_ratio = 0.5 + 0.3 * (position - 0.7) / 0.3  # 50-80%
            
            # Adjust based on alpha/beta - collect less when penalty is high
            if self.problem.alpha > 1.5 or self.problem.beta > 1.5:
                gold_ratio *= 0.5
            elif self.problem.alpha > 1.0 or self.problem.beta > 1.0:
                gold_ratio *= 0.7
            
            # Adjust based on current weight - collect less if already heavy
            if current_weight > weight_threshold * 0.5:
                gold_ratio *= 0.6
            
            # Adjust based on distance to depot - collect more if close
            if depot_dist < 3.0:
                gold_ratio *= 1.2
            elif depot_dist > 10.0:
                gold_ratio *= 0.8
            
            gold_ratio = np.clip(gold_ratio, 0.05, 0.85)
            gold_to_collect = max_gold * gold_ratio
            
            # Return to depot if weight is high
            # More aggressive returns when alpha/beta are high
            if (current_weight > weight_threshold and 
                i < len(improved_tour) - 2 and  # Don't return if almost done
                depot_dist < 8.0):  # Only if depot is reasonably close
                solution.append((0, 0))
                current_weight = 0.0
            
            solution.append((city, gold_to_collect))
            current_weight += gold_to_collect
        
        solution.append((0, 0))
        return solution
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        """Run conservative optimization"""
        self.start_time = time.time()
        self.log("Starting Conservative optimization")
        
        solution = self.generate_initial_solution()
        cost = evaluate_solution(self.problem, solution)
        
        self.update_best(solution, cost)
        self.log(f"Conservative solution cost: {cost:.2f}")
        
        return self.best_solution, self.best_cost

