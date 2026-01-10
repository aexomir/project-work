def generate_test_cases():
    test_cases = []
    
    # Tests 1-8: Size/Seed variations
    # 10 and 20 cities with 4 different seeds (42, 123, 456, 789)
    # α=1.0, β=1.5, density=0.5
    for num_cities in [10, 20]:
        for seed in [42, 123, 456, 789]:
            test_cases.append({
                'num_cities': num_cities,
                'alpha': 1.0,
                'beta': 1.5,
                'density': 0.5,
                'seed': seed
            })
    
    # Tests 9-12: Density variations
    # Test 9: 10 cities, α=1.0, β=1.5, density=0.3, seed=42
    test_cases.append({
        'num_cities': 10,
        'alpha': 1.0,
        'beta': 1.5,
        'density': 0.3,
        'seed': 42
    })
    # Test 10: 20 cities, α=1.0, β=1.5, density=0.3, seed=42
    test_cases.append({
        'num_cities': 20,
        'alpha': 1.0,
        'beta': 1.5,
        'density': 0.3,
        'seed': 42
    })
    # Test 11: 10 cities, α=1.0, β=1.5, density=0.7, seed=42
    test_cases.append({
        'num_cities': 10,
        'alpha': 1.0,
        'beta': 1.5,
        'density': 0.7,
        'seed': 42
    })
    # Test 12: 20 cities, α=1.0, β=1.5, density=0.7, seed=42
    test_cases.append({
        'num_cities': 20,
        'alpha': 1.0,
        'beta': 1.5,
        'density': 0.7,
        'seed': 42
    })
    
    # Tests 13-26: Parameter sweeps for different city sizes
    # Test 13-16: 5 cities with β=1.0, 1.2, 1.5, 2.0
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({
            'num_cities': 5,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 17-19: 10 cities with β=1.0, 1.2, 2.0 (skip 1.5, already in tests 1-8)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({
            'num_cities': 10,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 20-23: 15 cities with β=1.0, 1.2, 1.5, 2.0
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({
            'num_cities': 15,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 24-26: 20 cities with β=1.0, 1.2, 2.0 (skip 1.5, already in tests 1-8)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({
            'num_cities': 20,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Tests 27-30: α=2.0, β=2.0 for 5, 10, 15, 20 cities
    for num_cities in [5, 10, 15, 20]:
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 2.0,
            'beta': 2.0,
            'density': 0.5,
            'seed': 42
        })
    
    # Tests 31-42: Larger sizes (30, 50, 75, 100 cities)
    # For each size: β=1.0, β=1.5, α=2.0 β=2.0
    for num_cities in [30, 50, 75, 100]:
        # β=1.0
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 1.0,
            'beta': 1.0,
            'density': 0.5,
            'seed': 42
        })
        # β=1.5
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 1.0,
            'beta': 1.5,
            'density': 0.5,
            'seed': 42
        })
        # α=2.0, β=2.0
        test_cases.append({
            'num_cities': num_cities,
            'alpha': 2.0,
            'beta': 2.0,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 43: 50 cities, α=1.0, β=1.5, seed=123 (different seed)
    test_cases.append({
        'num_cities': 50,
        'alpha': 1.0,
        'beta': 1.5,
        'density': 0.5,
        'seed': 123
    })
    
    
    assert len(test_cases) == 43, f"Expected 43 test cases, got {len(test_cases)}"
    
    
    for i, test_case in enumerate(test_cases, 1):
        test_case['id'] = i
    
    return test_cases


test_cases = generate_test_cases()


if __name__ == "__main__":
    
    cases = generate_test_cases()
    print(f"Generated {len(cases)} test cases:")
    print()
    print(f"{'ID':<4} {'Cities':<7} {'Alpha':<6} {'Beta':<6} {'Density':<8} {'Seed':<6}")
    print("-" * 50)
    for case in cases:
        print(f"{case['id']:<4} {case['num_cities']:<7} {case['alpha']:<6.1f} "
              f"{case['beta']:<6.1f} {case['density']:<8.1f} {case['seed']:<6}")
    
    print()
    print(f"Total: {len(cases)} test cases")
