"""
Solution evaluation and validation for Traveling Thief Problem
"""

import networkx as nx
from typing import List, Tuple


def calculate_path_cost(problem, path: List[int], initial_weight: float) -> Tuple[float, float]:
    """
    Calculate cost for a path segment with given initial weight.
    
    Args:
        problem: Problem instance
        path: List of cities in order (e.g., [0, 3, 5, 0])
        initial_weight: Weight carried at start of path
    
    Returns:
        (total_cost, final_weight)
    """
    if len(path) < 2:
        return 0.0, initial_weight
    
    total_cost = 0.0
    current_weight = initial_weight
    
    for i in range(len(path) - 1):
        c1, c2 = path[i], path[i + 1]
        
        # Get distance between cities
        if not problem._graph.has_edge(c1, c2):
            # No direct edge - use shortest path
            try:
                shortest_path = nx.shortest_path(problem._graph, c1, c2, weight='dist')
                dist = nx.path_weight(problem._graph, shortest_path, weight='dist')
            except nx.NetworkXNoPath:
                return float('inf'), current_weight
        else:
            dist = problem._graph.edges[c1, c2]['dist']
        
        # Calculate cost with current weight
        cost = dist + (problem.alpha * dist * current_weight) ** problem.beta
        total_cost += cost
    
    return total_cost, current_weight


def evaluate_solution(problem, solution: List[Tuple[int, float]]) -> float:
    """
    Evaluate the total cost of a solution.
    
    Args:
        problem: Problem instance
        solution: List of (city, gold) tuples, e.g., [(3, 100), (5, 50), (0, 0), ...]
    
    Returns:
        Total cost of the solution
    """
    if not solution or solution[-1] != (0, 0):
        return float('inf')
    
    total_cost = 0.0
    current_weight = 0.0
    current_city = 0
    
    for city, gold in solution:
        if city == 0 and current_city != 0:
            # Return to depot - calculate cost with current weight
            cost, _ = calculate_path_cost(problem, [current_city, 0], current_weight)
            total_cost += cost
            current_weight = 0.0  # Unload at depot
            current_city = 0
        elif city != 0:
            # Moving to a new city
            if current_city != 0:
                # Not starting from depot
                cost, _ = calculate_path_cost(problem, [current_city, city], current_weight)
                total_cost += cost
            else:
                # Starting from depot
                cost, _ = calculate_path_cost(problem, [0, city], current_weight)
                total_cost += cost
            
            # Collect gold at this city
            current_weight += gold
            current_city = city
    
    return total_cost


def validate_solution(problem, solution: List[Tuple[int, float]]) -> Tuple[bool, str]:
    """
    Validate if a solution is valid.
    
    Args:
        problem: Problem instance
        solution: List of (city, gold) tuples
    
    Returns:
        (is_valid, error_message)
    """
    if not solution:
        return False, "Solution is empty"
    
    # Check starts at depot (implicitly - first move should be to a non-depot city or depot)
    # Check ends at depot
    if solution[-1] != (0, 0):
        return False, "Solution must end at depot (0, 0)"
    
    # Track which cities have been visited (excluding depot)
    visited_cities = set()
    num_cities = len(problem._graph.nodes)
    
    for city, gold in solution:
        if city == 0:
            # Depot visit
            if gold != 0:
                return False, f"Cannot collect gold at depot, got gold={gold}"
        else:
            # Regular city
            if city < 0 or city >= num_cities:
                return False, f"Invalid city index: {city}"
            
            # Check if city already visited
            if city in visited_cities:
                return False, f"City {city} visited multiple times"
            
            visited_cities.add(city)
            
            # Check gold amount is valid
            max_gold = problem._graph.nodes[city]['gold']
            if gold < 0 or gold > max_gold:
                return False, f"Invalid gold amount at city {city}: {gold} (max: {max_gold})"
    
    # Check all cities visited (excluding depot which is city 0)
    expected_cities = set(range(1, num_cities))
    if visited_cities != expected_cities:
        missing = expected_cities - visited_cities
        return False, f"Not all cities visited. Missing: {missing}"
    
    return True, "Valid solution"

