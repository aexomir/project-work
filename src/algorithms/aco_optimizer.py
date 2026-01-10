import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import time

from algorithms.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution


class ACOOptimizer(BaseOptimizer):
    def __init__(self, problem, max_iterations: int = 200, verbose: bool = False, seed: int = None,
                 num_ants: int = 50, alpha: float = 1.0, beta: float = 2.0, 
                 evaporation_rate: float = 0.1, q: float = 100.0):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit factor
        
        self.num_cities = len(problem._graph.nodes)
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        self.pheromone_gold = np.ones(self.num_cities)
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        return self.construct_solution()
    
    def get_heuristic_value(self, from_city: int, to_city: int, current_weight: float) -> float:
        if from_city == to_city:
            return 0.0
        
        if self.problem._graph.has_edge(from_city, to_city):
            dist = self.problem._graph.edges[from_city, to_city]['dist']
        else:
            try:
                path = nx.shortest_path(self.problem._graph, from_city, to_city, weight='dist')
                dist = nx.path_weight(self.problem._graph, path, weight='dist')
            except nx.NetworkXNoPath:
                return 0.0
        
        if dist == 0:
            return 0.0
        
        gold = self.problem._graph.nodes[to_city]['gold']
        
        weight_penalty = 1.0 + current_weight * self.problem.alpha * 0.01
        
        heuristic = (gold / dist) / weight_penalty
        return max(heuristic, 0.001)
    
    def construct_solution(self) -> List[Tuple[int, float]]:
        solution = []
        current_city = 0
        unvisited = set(range(1, self.num_cities))
        current_weight = 0.0
        
        while unvisited:
            probabilities = []
            cities = []
            
            for city in unvisited:
                pheromone = self.pheromone[current_city, city]
                heuristic = self.get_heuristic_value(current_city, city, current_weight)
                
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)
                cities.append(city)
            
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                probabilities = [1.0 / len(probabilities)] * len(probabilities)
            
            next_city = self.rng.choice(cities, p=probabilities)
            unvisited.remove(next_city)
            
            max_gold = self.problem._graph.nodes[next_city]['gold']
            gold_pheromone = self.pheromone_gold[next_city]
            
            base_ratio = gold_pheromone / (1.0 + gold_pheromone)
            weight_factor = max(0.2, 1.0 - current_weight / 100.0)
            gold_ratio = base_ratio * weight_factor
            gold_ratio = np.clip(gold_ratio, 0.1, 1.0)
            
            gold_collected = max_gold * gold_ratio
            
            if current_weight > 50.0 and len(unvisited) > len(cities) * 0.3:
                solution.append((0, 0))
                current_weight = 0.0
                current_city = 0
            
            solution.append((next_city, gold_collected))
            current_weight += gold_collected
            current_city = next_city
        
        solution.append((0, 0))
        return solution
    
    def update_pheromones(self, solutions: List[List[Tuple[int, float]]], costs: List[float]):
        self.pheromone *= (1.0 - self.evaporation_rate)
        self.pheromone_gold *= (1.0 - self.evaporation_rate)
        
        for solution, cost in zip(solutions, costs):
            if cost == float('inf'):
                continue
            
            deposit = self.q / cost
            
            tour = [0] + [city for city, _ in solution if city != 0] + [0]
            for i in range(len(tour) - 1):
                self.pheromone[tour[i], tour[i+1]] += deposit
                self.pheromone[tour[i+1], tour[i]] += deposit  # Symmetric
            
            for city, gold in solution:
                if city != 0:
                    max_gold = self.problem._graph.nodes[city]['gold']
                    gold_ratio = gold / max_gold
                    self.pheromone_gold[city] += deposit * gold_ratio
        
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
        self.pheromone_gold = np.clip(self.pheromone_gold, 0.01, 10.0)
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        self.start_time = time.time()
        self.log("Starting Ant Colony Optimization")
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            solutions = []
            costs = []
            
            for ant in range(self.num_ants):
                solution = self.construct_solution()
                cost = evaluate_solution(self.problem, solution)
                solutions.append(solution)
                costs.append(cost)
                
                if cost < self.best_cost:
                    self.update_best(solution, cost)
            
            self.update_pheromones(solutions, costs)
            
            if iteration % 20 == 0:
                avg_cost = np.mean([c for c in costs if c != float('inf')])
                self.log(f"Iteration {iteration}: Best={self.best_cost:.2f}, Avg={avg_cost:.2f}")
        
        self.log(f"Final best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost

