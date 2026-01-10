"""
Helper functions for TTP optimization
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict


def nearest_neighbor_tour(problem, start_city: int = 0, randomize: bool = False, rng=None) -> List[int]:
    """
    Generate a tour using nearest neighbor heuristic.
    
    Args:
        problem: Problem instance
        start_city: Starting city (usually 0)
        randomize: If True, add randomness to neighbor selection
        rng: Random number generator
    
    Returns:
        List of cities in tour order (excluding return to depot)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_cities = len(problem._graph.nodes)
    unvisited = set(range(1, num_cities))  # Exclude depot
    tour = []
    current = start_city
    
    while unvisited:
        # Find nearest unvisited city
        nearest_dist = float('inf')
        nearest_city = None
        candidates = []
        
        for city in unvisited:
            try:
                if problem._graph.has_edge(current, city):
                    dist = problem._graph.edges[current, city]['dist']
                else:
                    # Use shortest path distance
                    path = nx.shortest_path(problem._graph, current, city, weight='dist')
                    dist = nx.path_weight(problem._graph, path, weight='dist')
                
                if randomize:
                    candidates.append((city, dist))
                else:
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_city = city
            except nx.NetworkXNoPath:
                continue
        
        if randomize and candidates:
            # Choose probabilistically - prefer closer cities
            candidates.sort(key=lambda x: x[1])
            # Take top 5 candidates and choose randomly with bias towards closer
            top_candidates = candidates[:min(5, len(candidates))]
            weights = [1.0 / (1.0 + dist) for _, dist in top_candidates]
            weights = np.array(weights) / sum(weights)
            idx = rng.choice(len(top_candidates), p=weights)
            nearest_city = top_candidates[idx][0]
        
        if nearest_city is None:
            # Fallback: pick any unvisited city
            nearest_city = unvisited.pop()
            tour.append(nearest_city)
            current = nearest_city
        else:
            unvisited.remove(nearest_city)
            tour.append(nearest_city)
            current = nearest_city
    
    return tour


def two_opt_improve(problem, tour: List[int], max_iterations: int = 100) -> List[int]:
    """
    Improve a tour using 2-opt local search.
    
    Args:
        problem: Problem instance
        tour: List of cities (excluding depot)
        max_iterations: Maximum number of improvement iterations
    
    Returns:
        Improved tour
    """
    improved_tour = tour.copy()
    improved = True
    iteration = 0
    
    def tour_distance(t):
        """Calculate total distance of tour starting and ending at depot"""
        dist = 0.0
        # From depot to first city
        path = [0] + t + [0]
        for i in range(len(path) - 1):
            c1, c2 = path[i], path[i + 1]
            if problem._graph.has_edge(c1, c2):
                dist += problem._graph.edges[c1, c2]['dist']
            else:
                try:
                    sp = nx.shortest_path(problem._graph, c1, c2, weight='dist')
                    dist += nx.path_weight(problem._graph, sp, weight='dist')
                except nx.NetworkXNoPath:
                    return float('inf')
        return dist
    
    best_dist = tour_distance(improved_tour)
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(improved_tour) - 1):
            for j in range(i + 2, len(improved_tour)):
                # Try reversing segment between i and j
                new_tour = improved_tour[:i+1] + improved_tour[i+1:j+1][::-1] + improved_tour[j+1:]
                new_dist = tour_distance(new_tour)
                
                if new_dist < best_dist:
                    improved_tour = new_tour
                    best_dist = new_dist
                    improved = True
                    break
            
            if improved:
                break
    
    return improved_tour


def extract_route_stats(problem, solution: List[Tuple[int, float]]) -> Dict:
    """
    Extract statistics from a solution.
    
    Args:
        problem: Problem instance
        solution: List of (city, gold) tuples
    
    Returns:
        Dictionary with statistics
    """
    total_distance = 0.0
    total_gold = 0.0
    depot_returns = 0
    cities_visited = set()
    current_city = 0
    
    for city, gold in solution:
        if city == 0 and current_city != 0:
            depot_returns += 1
            # Calculate distance back to depot
            if problem._graph.has_edge(current_city, 0):
                total_distance += problem._graph.edges[current_city, 0]['dist']
            else:
                try:
                    path = nx.shortest_path(problem._graph, current_city, 0, weight='dist')
                    total_distance += nx.path_weight(problem._graph, path, weight='dist')
                except nx.NetworkXNoPath:
                    pass
            current_city = 0
        elif city != 0:
            # Moving to new city
            if current_city != 0:
                if problem._graph.has_edge(current_city, city):
                    total_distance += problem._graph.edges[current_city, city]['dist']
                else:
                    try:
                        path = nx.shortest_path(problem._graph, current_city, city, weight='dist')
                        total_distance += nx.path_weight(problem._graph, path, weight='dist')
                    except nx.NetworkXNoPath:
                        pass
            else:
                # From depot
                if problem._graph.has_edge(0, city):
                    total_distance += problem._graph.edges[0, city]['dist']
                else:
                    try:
                        path = nx.shortest_path(problem._graph, 0, city, weight='dist')
                        total_distance += nx.path_weight(problem._graph, path, weight='dist')
                    except nx.NetworkXNoPath:
                        pass
            
            total_gold += gold
            cities_visited.add(city)
            current_city = city
    
    # Calculate average gold collection ratio
    avg_gold_ratio = 0.0
    if len(cities_visited) > 0:
        total_available = sum(problem._graph.nodes[c]['gold'] for c in cities_visited)
        avg_gold_ratio = total_gold / total_available if total_available > 0 else 0.0
    
    return {
        'total_distance': total_distance,
        'total_gold_collected': total_gold,
        'depot_returns': depot_returns,
        'cities_visited': len(cities_visited),
        'avg_gold_ratio': avg_gold_ratio
    }

