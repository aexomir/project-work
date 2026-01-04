"""
Genetic Algorithm optimizer for TTP
"""

import numpy as np
from typing import List, Tuple
import time

from optimizers.base_optimizer import BaseOptimizer
from utils.evaluator import evaluate_solution
from utils.helpers import nearest_neighbor_tour, two_opt_improve


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm for TTP"""
    
    def __init__(self, problem, max_iterations: int = 500, verbose: bool = False, seed: int = None,
                 population_size: int = 150, mutation_rate: float = 0.15, crossover_rate: float = 0.8):
        super().__init__(problem, max_iterations, verbose, seed)
        self.rng = np.random.default_rng(seed)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness = []
    
    def generate_initial_solution(self) -> List[Tuple[int, float]]:
        """Generate a random initial solution"""
        # Get a tour
        tour = nearest_neighbor_tour(self.problem, randomize=True, rng=self.rng)
        
        # Assign random gold ratios
        solution = []
        for city in tour:
            max_gold = self.problem._graph.nodes[city]['gold']
            gold_ratio = self.rng.random()
            solution.append((city, max_gold * gold_ratio))
        
        solution.append((0, 0))
        return solution
    
    def initialize_population(self):
        """Create initial population"""
        self.log("Initializing population")
        self.population = []
        self.fitness = []
        
        for i in range(self.population_size):
            solution = self.generate_initial_solution()
            cost = evaluate_solution(self.problem, solution)
            self.population.append(solution)
            self.fitness.append(cost)
            
            if cost < self.best_cost:
                self.update_best(solution, cost)
    
    def tournament_selection(self, tournament_size: int = 5) -> List[Tuple[int, float]]:
        """Select parent using tournament selection"""
        indices = self.rng.choice(len(self.population), size=tournament_size, replace=False)
        best_idx = min(indices, key=lambda i: self.fitness[i])
        return self.population[best_idx].copy()
    
    def order_crossover(self, parent1: List[Tuple[int, float]], parent2: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Order crossover for tour + arithmetic crossover for gold"""
        # Extract tours (excluding depot returns)
        tour1 = [city for city, _ in parent1 if city != 0]
        tour2 = [city for city, _ in parent2 if city != 0]
        
        # Extract gold ratios
        gold_dict1 = {city: gold / self.problem._graph.nodes[city]['gold'] 
                     for city, gold in parent1 if city != 0}
        gold_dict2 = {city: gold / self.problem._graph.nodes[city]['gold'] 
                     for city, gold in parent2 if city != 0}
        
        # Order crossover for tour
        size = len(tour1)
        start, end = sorted(self.rng.choice(size, 2, replace=False))
        
        child_tour = [None] * size
        child_tour[start:end] = tour1[start:end]
        
        # Fill remaining positions from parent2
        pos = end
        for city in tour2:
            if city not in child_tour:
                if pos >= size:
                    pos = 0
                while child_tour[pos] is not None:
                    pos += 1
                    if pos >= size:
                        pos = 0
                child_tour[pos] = city
        
        # Arithmetic crossover for gold ratios
        alpha = self.rng.random()
        child_gold = {}
        for city in child_tour:
            if city in gold_dict1 and city in gold_dict2:
                child_gold[city] = alpha * gold_dict1[city] + (1 - alpha) * gold_dict2[city]
            elif city in gold_dict1:
                child_gold[city] = gold_dict1[city]
            else:
                child_gold[city] = gold_dict2[city]
        
        # Build solution
        solution = []
        for city in child_tour:
            max_gold = self.problem._graph.nodes[city]['gold']
            gold = max_gold * np.clip(child_gold[city], 0.0, 1.0)
            solution.append((city, gold))
        
        solution.append((0, 0))
        return solution
    
    def mutate(self, solution: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Apply mutation operators"""
        mutated = solution.copy()
        
        # Remove depot returns for manipulation
        tour = [(city, gold) for city, gold in mutated if city != 0]
        
        # Mutation 1: Swap two cities (50% chance)
        if self.rng.random() < 0.5 and len(tour) > 1:
            i, j = self.rng.choice(len(tour), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        
        # Mutation 2: Adjust gold ratios (70% chance)
        if self.rng.random() < 0.7:
            for i in range(len(tour)):
                if self.rng.random() < 0.3:  # Mutate 30% of cities
                    city, gold = tour[i]
                    max_gold = self.problem._graph.nodes[city]['gold']
                    current_ratio = gold / max_gold
                    # Add random perturbation
                    new_ratio = current_ratio + self.rng.normal(0, 0.2)
                    new_ratio = np.clip(new_ratio, 0.0, 1.0)
                    tour[i] = (city, max_gold * new_ratio)
        
        # Mutation 3: Insert depot return (10% chance)
        if self.rng.random() < 0.1 and len(tour) > 2:
            insert_pos = self.rng.integers(1, len(tour))
            tour.insert(insert_pos, (0, 0))
        
        # Rebuild solution
        mutated = tour + [(0, 0)]
        return mutated
    
    def optimize(self) -> Tuple[List[Tuple[int, float]], float]:
        """Run genetic algorithm"""
        self.start_time = time.time()
        self.log("Starting Genetic Algorithm")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.max_iterations):
            self.iteration = generation
            
            new_population = []
            new_fitness = []
            
            # Elitism: keep top 10%
            elite_count = max(1, self.population_size // 10)
            elite_indices = np.argsort(self.fitness)[:elite_count]
            for idx in elite_indices:
                new_population.append(self.population[idx])
                new_fitness.append(self.fitness[idx])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                if self.rng.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if self.rng.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                # Evaluate
                cost = evaluate_solution(self.problem, child)
                new_population.append(child)
                new_fitness.append(cost)
                
                if cost < self.best_cost:
                    self.update_best(child, cost)
            
            self.population = new_population
            self.fitness = new_fitness
            
            # Log progress
            if generation % 50 == 0:
                avg_fitness = np.mean(self.fitness)
                self.log(f"Generation {generation}: Best={self.best_cost:.2f}, Avg={avg_fitness:.2f}")
        
        self.log(f"Final best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost

