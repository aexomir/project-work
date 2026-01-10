def generate_test_cases():
    """
    Generate 43 test cases.
    
    Returns:
        List of test case configuration dictionaries
    """
    test_cases = []
    
    # 1-8: Size/Seed variations (10 and 20 cities with 4 different seeds)
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
    
    # 9-12: Density variations
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
    
    # 13-26: Parameter sweeps for 5, 10, 15, 20 cities
    # Test 13-16: 5 cities with various β
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({
            'num_cities': 5,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 17-19: 10 cities with β=1.0, 1.2, 2.0 (skip 1.5, already in 1-8)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({
            'num_cities': 10,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 20-22: 15 cities with β=1.0, 1.2, 1.5, 2.0
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({
            'num_cities': 15,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 23-25: 20 cities with β=1.0, 1.2, 2.0 (skip 1.5, already in 1-8)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({
            'num_cities': 20,
            'alpha': 1.0,
            'beta': beta,
            'density': 0.5,
            'seed': 42
        })
    
    # Test 26: 20 cities, α=1.0, β=2.0 (already added above, so we need to adjust)
    # Actually, test 25 is the last one, so test 26 should be something else
    # Looking at the table, test 26 is 20 cities, α=1.0, β=2.0 - but that's test 25
    # Wait, let me recount. After test 25 (20 cities, β=2.0), we have:
    # Test 26: 20 cities, α=1.0, β=2.0 - but that's a duplicate
    # Actually looking at the example, test 26 is 20 cities, α=1.0, β=2.0 which we already have
    # Let me check the table more carefully - test 26 shows 20 cities, α=1.0, β=2.0
    
    # Actually, I think the issue is that tests 13-26 include α=2.0 cases too
    # Let me look at tests 27-30: they are α=2.0, β=2.0 for 5, 10, 15, 20
    
    # So tests 13-26 should be:
    # 13-16: 5 cities, α=1.0, β=1.0, 1.2, 1.5, 2.0 (4 cases)
    # 17-19: 10 cities, α=1.0, β=1.0, 1.2, 2.0 (3 cases, skip 1.5)
    # 20-22: 15 cities, α=1.0, β=1.0, 1.2, 1.5, 2.0 (4 cases) - wait that's 5 cases
    # Let me recount from the table:
    
    # Test 13: 5 cities, α=1.0, β=1.0
    # Test 14: 5 cities, α=1.0, β=1.2
    # Test 15: 5 cities, α=1.0, β=1.5
    # Test 16: 5 cities, α=1.0, β=2.0
    # Test 17: 10 cities, α=1.0, β=1.0
    # Test 18: 10 cities, α=1.0, β=1.2
    # Test 19: 10 cities, α=1.0, β=2.0
    # Test 20: 15 cities, α=1.0, β=1.0
    # Test 21: 15 cities, α=1.0, β=1.2
    # Test 22: 15 cities, α=1.0, β=1.5
    # Test 23: 15 cities, α=1.0, β=2.0
    # Test 24: 20 cities, α=1.0, β=1.0
    # Test 25: 20 cities, α=1.0, β=1.2
    # Test 26: 20 cities, α=1.0, β=2.0
    
    # So that's: 4 + 3 + 4 + 3 = 14 cases for tests 13-26
    
    # But I already generated some of these. Let me fix:
    # Remove the duplicates I added and rebuild correctly
    
    # Actually, I think I need to rebuild from scratch more carefully
    # Let me just manually list all 43 based on the table
    pass  # I'll rebuild this properly
    
    # Let me try a different approach - build the list manually based on exact table
    test_cases = []
    
    # Tests 1-8: 10 and 20 cities, seeds 42,123,456,789, α=1.0, β=1.5, density=0.5
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
    test_cases.append({'num_cities': 10, 'alpha': 1.0, 'beta': 1.5, 'density': 0.3, 'seed': 42})
    test_cases.append({'num_cities': 20, 'alpha': 1.0, 'beta': 1.5, 'density': 0.3, 'seed': 42})
    test_cases.append({'num_cities': 10, 'alpha': 1.0, 'beta': 1.5, 'density': 0.7, 'seed': 42})
    test_cases.append({'num_cities': 20, 'alpha': 1.0, 'beta': 1.5, 'density': 0.7, 'seed': 42})
    
    # Tests 13-26: Parameter sweeps
    # 5 cities: β=1.0, 1.2, 1.5, 2.0
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({'num_cities': 5, 'alpha': 1.0, 'beta': beta, 'density': 0.5, 'seed': 42})
    # 10 cities: β=1.0, 1.2, 2.0 (skip 1.5)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({'num_cities': 10, 'alpha': 1.0, 'beta': beta, 'density': 0.5, 'seed': 42})
    # 15 cities: β=1.0, 1.2, 1.5, 2.0
    for beta in [1.0, 1.2, 1.5, 2.0]:
        test_cases.append({'num_cities': 15, 'alpha': 1.0, 'beta': beta, 'density': 0.5, 'seed': 42})
    # 20 cities: β=1.0, 1.2, 2.0 (skip 1.5)
    for beta in [1.0, 1.2, 2.0]:
        test_cases.append({'num_cities': 20, 'alpha': 1.0, 'beta': beta, 'density': 0.5, 'seed': 42})
    
    # Tests 27-30: α=2.0, β=2.0 for 5, 10, 15, 20
    for num_cities in [5, 10, 15, 20]:
        test_cases.append({'num_cities': num_cities, 'alpha': 2.0, 'beta': 2.0, 'density': 0.5, 'seed': 42})
    
    # Tests 31-43: Larger sizes
    # 30, 50, 75, 100 cities with various α/β
    for num_cities in [30, 50, 75, 100]:
        # β=1.0
        test_cases.append({'num_cities': num_cities, 'alpha': 1.0, 'beta': 1.0, 'density': 0.5, 'seed': 42})
        # β=1.5
        test_cases.append({'num_cities': num_cities, 'alpha': 1.0, 'beta': 1.5, 'density': 0.5, 'seed': 42})
        # β=2.0
        test_cases.append({'num_cities': num_cities, 'alpha': 1.0, 'beta': 2.0, 'density': 0.5, 'seed': 42})
        # α=2.0, β=2.0
        test_cases.append({'num_cities': num_cities, 'alpha': 2.0, 'beta': 2.0, 'density': 0.5, 'seed': 42})
    
    # Test 43: 50 cities, α=1.0, β=1.5, seed=123
    # But wait, we already have 50 cities with those params in test 35
    # Looking at the table, test 43 is 50 cities, α=1.0, β=1.5, seed=123
    # So we need to replace the last 50 cities case or add it
    # Actually, test 35 is 50 cities, α=1.0, β=1.5, seed=42
    # Test 43 is 50 cities, α=1.0, β=1.5, seed=123
    # So we need to remove one of the 50 cities cases and add the seed=123 one
    
    # Let me count: we have 4 city sizes × 4 cases = 16 cases for tests 31-46
    # But we only need tests 31-43, which is 13 cases
    # So for 30, 50, 75, 100: we need 3, 3, 3, 4 cases = 13 total
    
    # Let me rebuild tests 31-43 more carefully:
    # Remove the last batch and rebuild
    # We had 8 + 4 + 14 + 4 = 30 cases so far
    # Need 13 more for 31-43
    
    # Actually, let me just remove everything after test 30 and rebuild 31-43
    test_cases = test_cases[:30]  # Keep first 30
    
    # Tests 31-43: 
    # 30 cities: β=1.0, 1.5, 2.0, α=2.0β=2.0 (4 cases)
    # 50 cities: β=1.0, 1.5, 2.0 (3 cases) 
    # 75 cities: β=1.0, 1.5, 2.0, α=2.0β=2.0 (4 cases)
    # 100 cities: β=1.0, 1.5, 2.0 (3 cases)
    # Total: 4+3+4+3 = 14, but we need 13
    
    # Looking at table again:
    # Test 31: 30 cities, α=1.0, β=1.0
    # Test 32: 30 cities, α=1.0, β=1.5
    # Test 33: 30 cities, α=2.0, β=2.0
    # Test 34: 50 cities, α=1.0, β=1.0
    # Test 35: 50 cities, α=1.0, β=1.5
    # Test 36: 50 cities, α=2.0, β=2.0
    # Test 37: 75 cities, α=1.0, β=1.0
    # Test 38: 75 cities, α=1.0, β=1.5
    # Test 39: 75 cities, α=2.0, β=2.0
    # Test 40: 100 cities, α=1.0, β=1.0
    # Test 41: 100 cities, α=1.0, β=1.5
    # Test 42: 100 cities, α=2.0, β=2.0
    # Test 43: 50 cities, α=1.0, β=1.5, seed=123
    
    # So: 30 cities: 3 cases, 50 cities: 3 cases (but test 43 is different seed), 75 cities: 3 cases, 100 cities: 3 cases
    # Total: 3+3+3+3 = 12, plus test 43 = 13
    
    for num_cities in [30, 50, 75, 100]:
        test_cases.append({'num_cities': num_cities, 'alpha': 1.0, 'beta': 1.0, 'density': 0.5, 'seed': 42})
        test_cases.append({'num_cities': num_cities, 'alpha': 1.0, 'beta': 1.5, 'density': 0.5, 'seed': 42})
        test_cases.append({'num_cities': num_cities, 'alpha': 2.0, 'beta': 2.0, 'density': 0.5, 'seed': 42})
    
    # Test 43: Replace the last 50 cities case (test 35 duplicate) with seed=123
    # Actually test 35 is the second 50 cities case, test 36 is third
    # Test 43 should be 50 cities, α=1.0, β=1.5, seed=123
    # So we replace test case index 34 (0-indexed) which is the second 50 cities case
    # Wait, let me count: tests 31-33 are 30 cities (3 cases)
    # Tests 34-36 are 50 cities (3 cases) - but test 43 wants to replace one
    # Actually, test 43 is a separate case with seed=123
    # So we have: 30 (3) + 50 (3) + 75 (3) + 100 (3) = 12, then test 43 makes 13
    
    # But we generated 12 cases above, so we need to add test 43
    # Actually, test 43 replaces one of the 50 cities cases
    # Looking at the table: test 35 is 50 cities α=1.0 β=1.5 seed=42
    # Test 43 is 50 cities α=1.0 β=1.5 seed=123
    # So we should have both - test 35 and test 43 are different
    
    # So we have 12 cases, and test 43 is the 13th
    # But wait, that means we'd have 30+12=42 cases, and test 43 makes 43
    
    # Let me just add test 43 as specified:
    test_cases.append({'num_cities': 50, 'alpha': 1.0, 'beta': 1.5, 'density': 0.5, 'seed': 123})
    
    # Ensure we have exactly 43 test cases
    assert len(test_cases) == 43, f"Expected 43 test cases, got {len(test_cases)}"
    
    # Add sequential IDs
    for i, test_case in enumerate(test_cases, 1):
        test_case['id'] = i
    
    return test_cases


# Export test_cases for direct import
test_cases = generate_test_cases()


if __name__ == "__main__":
    # Print test cases for verification
    cases = generate_test_cases()
    print(f"Generated {len(cases)} test cases:")
    print()
    for case in cases:
        print(f"ID {case['id']:2d}: {case['num_cities']:3d} cities, "
              f"α={case['alpha']:.1f}, β={case['beta']:.1f}, "
              f"density={case['density']:.1f}, seed={case['seed']}")
