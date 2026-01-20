import networkx as nx
from typing import List, Tuple


def calculate_path_cost(problem, path: List[int], initial_weight: float) -> Tuple[float, float]:
    if len(path) < 2:
        return 0.0, initial_weight
    
    total_cost = 0.0
    current_weight = initial_weight
    
    for i in range(len(path) - 1):
        c1, c2 = path[i], path[i + 1]
        
        
        if not problem._graph.has_edge(c1, c2):
            try:
                shortest_path = nx.shortest_path(problem._graph, c1, c2, weight='dist')
                dist = nx.path_weight(problem._graph, shortest_path, weight='dist')
            except nx.NetworkXNoPath:
                return float('inf'), current_weight
        else:
            dist = problem._graph.edges[c1, c2]['dist']
        
        
        cost = dist + (problem.alpha * dist * current_weight) ** problem.beta
        total_cost += cost
    
    return total_cost, current_weight


def evaluate_solution(problem, solution: List[Tuple[int, float]]) -> float:
    if not solution or solution[-1] != (0, 0):
        return float('inf')
    
    total_cost = 0.0
    current_weight = 0.0
    current_city = 0
    
    for city, gold in solution:
        if city == 0 and current_city != 0:
            
            cost, _ = calculate_path_cost(problem, [current_city, 0], current_weight)
            total_cost += cost
            current_weight = 0.0
            current_city = 0
        elif city != 0:
            
            if current_city != 0:
                
                cost, _ = calculate_path_cost(problem, [current_city, city], current_weight)
                total_cost += cost
            else:
                
                cost, _ = calculate_path_cost(problem, [0, city], current_weight)
                total_cost += cost
            
            
            current_weight += gold
            current_city = city
    
    return total_cost


def validate_solution(problem, solution: List[Tuple[int, float]]) -> Tuple[bool, str]:
    if not solution:
        return False, "Solution is empty"
    
    
    if solution[-1] != (0, 0):
        return False, "Solution must end at depot (0, 0)"
    
    visited_cities = set()
    num_cities = len(problem._graph.nodes)
    
    for city, gold in solution:
        if city == 0:
            if gold != 0:
                return False, f"Cannot collect gold at depot, got gold={gold}"
        else:
            if city < 0 or city >= num_cities:
                return False, f"Invalid city index: {city}"
            
            if city in visited_cities:
                return False, f"City {city} visited multiple times"
            
            visited_cities.add(city)
            
            
            max_gold = problem._graph.nodes[city]['gold']
            if gold < 0 or gold > max_gold:
                return False, f"Invalid gold amount at city {city}: {gold} (max: {max_gold})"
    
    
    expected_cities = set(range(1, num_cities))
    if visited_cities != expected_cities:
        missing = expected_cities - visited_cities
        return False, f"Not all cities visited. Missing: {missing}"
    
    return True, "Valid solution"

