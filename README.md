# Gold Collection Routing Optimization

**Student ID: 340808**

## Overview

This project implements an optimized solution for the **Gold Collection with Weight Penalty Problem** - a variant of the Traveling Thief Problem (TTP). The goal is to find the most cost-effective route for collecting gold from multiple cities while minimizing travel costs that increase exponentially with the weight of carried gold.

The main solution (`ConservativeOptimizer`) uses a conservative, trip-aware heuristic strategy that strategically manages gold collection and depot returns to avoid catastrophic weight penalties, achieving **88% average improvement** over the baseline across diverse problem configurations.

## Problem Description

A thief starts at a base city (city 0) and must collect gold from various cities in a graph while managing:

- **Distance costs**: Base travel cost between cities
- **Weight penalty**: Travel cost increases exponentially with carried gold weight
- **Trip planning**: Option to return to base multiple times to unload gold

### Problem Generation

The `Problem` class generates instances with:

- **Cities**: `num_cities` cities placed uniformly at random in `[0,1]²`, with city 0 fixed at `(0.5, 0.5)`
- **Gold distribution**: Each city (except city 0) has random gold in `[1, 1000]`
- **Graph connectivity**: Edges added probabilistically based on `density` parameter, plus guaranteed connections between consecutive cities to ensure connectivity
- **Cost parameters**: `alpha` (weight penalty multiplier) and `beta` (weight penalty exponent)

### Rules

1. The thief must start and finish at city 0
2. Returning to city 0 during the route is allowed to unload collected gold before continuing
3. Each city can be visited multiple times with partial gold collection
4. The solution must return a path in the format: `[(c₁, g₁), (c₂, g₂), ..., (cₙ, gₙ), (0, 0)]` where `cᵢ` is the city visited and `gᵢ` is the gold collected at that city

## Cost Function

For each edge traversed:

```
cost = distance + (α × distance × weight)^β
```

Where:

- **distance**: Edge distance in the graph (Euclidean distance between city coordinates)
- **weight**: Amount of gold currently carried
- **α (alpha)**: Weight penalty multiplier
- **β (beta)**: Weight penalty exponent

The baseline solution (`Problem.baseline()`) makes separate round trips to each city: it travels to each city with zero weight, then returns with full gold. This provides a simple but expensive reference point.

## Solution Approach

The implemented algorithm (`ConservativeOptimizer` in `src/algorithms/conservative_optimizer.py`) uses a conservative, multi-trip strategy with several optimization techniques:

### 1. Tour Construction & Smoothing

- **Nearest Neighbor Tour**: Builds an initial tour starting from city 0 using nearest-neighbor heuristics
- **2-Opt Local Search**: Applies 2-opt improvements (30 iterations) to reduce total tour distance before gold allocation
- This creates a low-distance visiting order that minimizes base travel costs

### 2. Position-Based Gold Collection Strategy

The algorithm uses a sophisticated position-aware gold collection ratio that varies based on the city's position in the tour:

- **Early cities (0-30% of tour)**: Collect 10-20% of available gold
  - Low collection prevents early weight accumulation
  - Allows flexibility for later optimization
  
- **Mid-tour cities (30-70% of tour)**: Collect 20-50% of available gold
  - Moderate collection balances opportunity with weight management
  
- **Late cities (70-100% of tour)**: Collect 50-80% of available gold
  - Higher collection ratios since depot return is imminent
  - Less penalty risk for remaining travel

### 3. Penalty-Aware Load Management

The algorithm dynamically adjusts collection ratios based on problem parameters:

- **High penalty scenarios** (α > 1.5 or β > 1.5): Collection ratios reduced by 50%
- **Moderate penalties** (α > 1.0 or β > 1.0): Collection ratios reduced by 30%
- **Weight threshold**: Base threshold of 15.0 units, scaled down by penalty factors
- **Current load adjustment**: If current weight exceeds 50% of threshold, collection ratios further reduced by 40%

This ensures the algorithm never accumulates dangerous weight levels that would cause exponential cost explosions.

### 4. Depot-Distance Adjustment

The algorithm considers proximity to the depot when deciding gold collection:

- **Near depot** (< 3.0 distance units): Collection ratio increased by 20%
  - Safe to collect more since depot return is cheap
  
- **Far from depot** (> 10.0 distance units): Collection ratio reduced by 20%
  - Conservative collection to avoid expensive return journey

### 5. Trip Splitting & Strategic Depot Returns

The algorithm inserts depot returns `(0, 0)` mid-tour when:

- Current weight exceeds the adaptive threshold
- At least 2 cities remain in the tour
- Depot is reasonably close (< 8.0 distance units)

This creates multiple trips, each optimized for its segment of the tour, preventing the exponential weight penalty from dominating costs.

### 6. Adaptive Threshold Calculation

The weight threshold adapts to problem difficulty:

```python
base_threshold = 15.0
weight_threshold = base_threshold / (1.0 + (α - 1.0) × 0.5 + (β - 1.0) × 0.5)
```

Higher α and β values result in lower thresholds, forcing more frequent depot returns and more conservative gold collection.

## Algorithm Flow

The following flowchart illustrates the Conservative Optimizer's decision-making process:

![Conservative Optimizer Algorithm Flow](algorithm_flow.png)

*Figure 1: Complete algorithm flow showing tour construction, gold collection ratio calculation, weight threshold checking, and strategic depot returns.*

## Key Features

### Advantages

- **Significant Cost Reduction**: Average **88.08% improvement** over baseline (algorithm comparison), **83.51% improvement** (conservative-only benchmarks)
- **Adaptive Strategy**: Adjusts behavior based on problem parameters (α, β) - more conservative when penalties are high
- **Scalability**: Handles problems from 5-100+ cities efficiently
- **Robustness**: Works well across diverse parameter ranges and graph densities
- **Guaranteed Safety**: Never accumulates dangerous weight levels that cause exponential cost explosions
- **100% Success Rate**: All test cases produce valid solutions

### Performance Highlights

Based on comprehensive testing:

**Algorithm Comparison** (16 test cases, 100 cities, varying density/α/β):
- **Conservative Optimizer**: 88.08% average improvement, 70.05% to 97.72% range, 100% success rate
- **Greedy Optimizer**: -176.42% average (worse than baseline) - demonstrates need for conservative strategy
- **Genetic Algorithm**: -1612.47% average (much worse) - struggles with weight penalty management

**Conservative-Only Benchmarks** (43 diverse test cases, 5-100 cities):
- **Average Improvement**: 83.51% over baseline
- **Best Case**: 97.29% improvement (36.92× speedup)
- **Worst Case**: 45.05% improvement (1.82× speedup) - still significantly better than baseline
- **Success Rate**: 100% (43/43 valid solutions)
- **Average Speedup**: 10.96× cost reduction
- **Execution Time**: Average 4.14s (ranging from 0.0001s for small problems to 27.77s for 100-city problems)

### When Does It Excel?

- **High weight penalties** (β ≥ 1.5): Near-perfect optimization (96-97% improvement)
- **High cost scaling** (α = 2.0): Exceptional performance (96-97% improvement)
- **Large graphs** (50-100 cities): Better strategic planning opportunities
- **Any graph density** (0.3-2.0): Consistent performance across sparse and dense graphs

### When Is It More Modest?

- **Low weight penalties** (β = 1.0): Still achieves 45-75% improvement, but less dramatic gains
- **Very small graphs** (< 10 cities): Limited routing options, but still beats baseline

## File Structure

```
project-work/
├── Problem.py                    # Problem generator class
├── s340808.py                   # Main solution entry point (uses ConservativeOptimizer)
├── base_requirements.txt        # Basic Python dependencies
├── requirements.txt             # Full dependencies (includes tqdm)
├── README.md                    # This file
├── outputs/
│   ├── alg_comparison_results.json           # Algorithm comparison results
│   └── conservative_comparison_results.json  # Conservative optimizer benchmarks
├── notebooks/
│   └── problem.ipynb           # Jupyter notebook for exploration
├── run_algorithm_comparison.sh  # Shell script for algorithm comparison
├── run_conservative_tests.sh    # Shell script for conservative benchmarks
├── run_large_scale_tests.sh     # Shell script for large-scale testing
└── src/
    ├── algorithms/
    │   ├── base_optimizer.py           # Base class for optimizers
    │   ├── conservative_optimizer.py   # Main solution (ConservativeOptimizer)
    │   ├── greedy_optimizer.py         # Greedy baseline
    │   ├── improved_greedy_optimizer.py
    │   ├── smart_optimizer.py
    │   ├── genetic_optimizer.py
    │   ├── simulated_annealing_optimizer.py
    │   ├── local_search_optimizer.py
    │   └── aco_optimizer.py
    ├── baseline/
    │   ├── baseline_impl.py    # Problem class (used in comparisons)
    │   └── run_baseline.py
    ├── utils/
    │   ├── evaluator.py         # Solution validation and evaluation
    │   └── helpers.py           # Tour construction utilities (NN, 2-opt)
    ├── compare_algorithms.py   # Algorithm comparison script
    ├── run_conservative_comparison.py  # Conservative benchmark script
    ├── generate_report.py       # Generate markdown reports from JSON results
    └── generate_test_cases.py  # Test case generator
```

## Usage

### Basic Usage

```python
from Problem import Problem
from src.algorithms.conservative_optimizer import ConservativeOptimizer

# Create a problem instance
problem = Problem(
    num_cities=50,
    alpha=2.0,
    beta=2.0,
    density=0.5,
    seed=42
)

# Create optimizer and solve
optimizer = ConservativeOptimizer(
    problem,
    max_iterations=1,
    verbose=True,
    seed=42
)

# Get the optimized path
solution, cost = optimizer.optimize()

# Path format: [(city, gold_amount), ...]
# Example: [(0, 0), (1, 150.5), (2, 0), (0, 0), (3, 200.0), (0, 0)]
```

### Using the Solution Entry Point

For the exam submission format:

```python
from Problem import Problem
from s340808 import solution

# Create a problem instance
problem = Problem(
    num_cities=50,
    alpha=2.0,
    beta=2.0,
    density=0.5,
    seed=42
)

# Get the optimized path
path = solution(problem)
```

### Running Performance Tests

**Algorithm Comparison** (compares Conservative vs Greedy vs Genetic):

```bash
python src/compare_algorithms.py
# Or use the shell script:
./run_algorithm_comparison.sh
```

This generates results in `outputs/alg_comparison_results.json`.

**Conservative Optimizer Benchmarks**:

```bash
python src/run_conservative_comparison.py
# Or use the shell script:
./run_conservative_tests.sh
```

This generates results in `outputs/conservative_comparison_results.json`.

**Large-Scale Tests**:

```bash
./run_large_scale_tests.sh
```

**Generate Detailed Report**:

```bash
python src/generate_report.py --input outputs/conservative_comparison_results.json --output outputs/conservative_comparison_report.md
```

This creates a comprehensive markdown report with detailed statistics, tables, and analysis.

## Implementation Details

### Path Format

The solution returns a list of tuples:

```
[(c₁, g₁), (c₂, g₂), ..., (cₙ, gₙ), (0, 0)]
```

Where:

- **cᵢ**: City visited at step i
- **gᵢ**: Gold amount collected at city cᵢ
- **(0, 0)**: Depot return (can appear multiple times mid-route or at the end)

### Key Functions

**`ConservativeOptimizer.generate_initial_solution()`**

Builds the complete solution:
1. Constructs nearest-neighbor tour
2. Applies 2-opt improvements
3. Computes depot distances for all cities
4. Iterates through tour, calculating gold collection ratios
5. Inserts depot returns when weight threshold exceeded
6. Returns complete path with gold allocations

**`ConservativeOptimizer.optimize()`**

Main optimization entry point:
- Calls `generate_initial_solution()`
- Evaluates solution cost using `evaluate_solution()`
- Returns `(solution, cost)` tuple

**Gold Ratio Calculation**

The gold collection ratio combines multiple factors:

```python
# Base ratio from position
if position < 0.3:
    gold_ratio = 0.1 + 0.1 * position / 0.3  # 10-20%
elif position < 0.7:
    gold_ratio = 0.2 + 0.3 * (position - 0.3) / 0.4  # 20-50%
else:
    gold_ratio = 0.5 + 0.3 * (position - 0.7) / 0.3  # 50-80%

# Apply penalty adjustments
if alpha > 1.5 or beta > 1.5:
    gold_ratio *= 0.5
elif alpha > 1.0 or beta > 1.0:
    gold_ratio *= 0.7

# Apply weight and depot adjustments
# ... (see conservative_optimizer.py for details)

# Clip to safe range
gold_ratio = np.clip(gold_ratio, 0.05, 0.85)
```

## Performance Results

### Algorithm Comparison Summary

Based on 16 test cases (100 cities, density ∈ {0.5, 1.0, 1.5, 2.0}, α ∈ {1, 2}, β ∈ {1, 2}):

| Algorithm | Avg Improvement | Min Improvement | Max Improvement | Success Rate | Avg Time |
|-----------|----------------|-----------------|-----------------|--------------|----------|
| **Conservative** | **88.08%** | **70.05%** | **97.72%** | **100%** | **7.85s** |
| Greedy | -176.42% | -578.71% | -8.75% | 100% | 0.27s |
| Genetic Algorithm | -1612.47% | -3288.49% | -306.62% | 100% | 86.73s |

**Key Insight**: The conservative strategy dramatically outperforms greedy and genetic approaches, which fail to manage weight penalties effectively.

### Conservative Optimizer Detailed Benchmarks

Based on 43 diverse test cases (5-100 cities, α: 1.0-2.0, β: 1.0-2.0, density: 0.3-0.7, seeds: 42-789):

| Metric | Value |
|--------|-------|
| **Average Improvement** | **83.51%** |
| **Best Improvement** | **97.29%** (36.92× speedup) |
| **Worst Improvement** | **45.05%** (1.82× speedup) |
| **Success Rate** | **100%** (43/43) |
| **Average Speedup** | **10.96×** |
| **Std Dev** | 11.11% |
| **Avg Execution Time** | 4.14s |

### Performance by Problem Size

| Cities | Avg Improvement | Avg Time | Observation |
|--------|----------------|----------|-------------|
| 5 | 82.96% | 0.0001s | Very fast |
| 10 | 73.25% | 0.002s | Fast |
| 15 | 81.25% | 0.022s | Quick |
| 20 | 85.62% | 0.037s | Quick |
| 30 | 85.38% | 0.181s | Acceptable |
| 50 | 86.61% | 5.62s | Good |
| 75 | 85.48% | 24.80s | Acceptable |
| 100 | 84.84% | 27.60s | Acceptable |

### Performance by Beta Parameter

| Beta (β) | Test Count | Avg Improvement | Description |
|----------|------------|-----------------|-------------|
| 1.0 | 8 | 68.57% | Low penalty - moderate improvement |
| 1.2 | 3 | 75.15% | Moderate penalty - good improvement |
| 1.5 | 20 | 85.62% | Strong penalty - **excellent improvement** |
| 2.0 | 12 | 95.12% | Extreme penalty - **near-perfect optimization** |

**Insight**: Higher β values amplify the benefits of conservative weight management, with β ≥ 1.5 showing exceptional results.

### Performance by Alpha Parameter

| Alpha (α) | Test Count | Avg Improvement | Best Case |
|-----------|------------|-----------------|-----------|
| 1.0 | 31 | 80.23% | 97.28% |
| 2.0 | 12 | 94.25% | **97.29%** |

**Insight**: Higher α values (cost scaling) further amplify the conservative strategy's advantages.

## Algorithm Complexity

- **Preprocessing**: O(V²) for all-pairs shortest paths computation (where V = number of cities)
- **Tour Construction**: O(V²) for nearest-neighbor tour + 2-opt improvements
- **Gold Allocation**: O(V) per tour iteration
- **Overall**: Approximately O(V²) per solution generation

The algorithm is efficient enough for real-time applications on problems up to 100 cities, with most practical problems (≤50 cities) completing in under 6 seconds.

## Design Decisions

### Why Conservative Instead of Aggressive?

The weight penalty grows exponentially with carried gold. Aggressive strategies (like greedy or genetic algorithms) often accumulate too much weight early, causing catastrophic cost explosions. The conservative approach prioritizes **safety** over exploitation, ensuring consistent performance across all scenarios.

### Why Position-Based Gold Ratios?

Early cities collect less gold to maintain flexibility. Late cities can collect more since depot return is imminent. This creates a natural "ramp-up" pattern that balances opportunity with risk.

### Why Adaptive Thresholds?

Different problem parameters (α, β) require different risk tolerance. High penalties demand lower thresholds and more frequent depot returns. The adaptive threshold ensures optimal behavior across diverse problem configurations.

### Why Multi-Trip Planning?

Making multiple trips with partial loads often costs less than carrying all gold at once when penalties are high. The algorithm automatically determines optimal trip boundaries based on weight accumulation and depot proximity.

### Why 2-Opt Before Gold Allocation?

Optimizing the tour distance first reduces base travel costs. Gold allocation then focuses on managing weight penalties, creating a clean separation of concerns.

## Future Improvements

Potential enhancements:

1. **Look-ahead heuristics**: Consider 2-3 cities ahead instead of just current position when deciding gold collection
2. **Dynamic threshold adjustment**: Vary weight threshold based on observed penalties during execution
3. **Local search refinement**: Post-process solution with gold reallocation heuristics (not just tour order)
4. **Hybrid strategies**: Combine conservative approach with smart optimizer ideas for specific scenarios
5. **Parallel trip evaluation**: Explore multiple trip strategies simultaneously
6. **Machine learning**: Learn optimal gold ratios from historical problem instances

## Testing

The project includes comprehensive test suites:

- **Algorithm Comparison**: Tests Conservative vs Greedy vs Genetic on 16 diverse 100-city problems
- **Conservative Benchmarks**: 43 test cases covering 5-100 cities, varying α/β/density/seeds
- **Large-Scale Tests**: Extended testing on 1000-city problems (optional, via `--large` flag)

All tests validate:
- Solution correctness (valid path format, starts/ends at depot)
- Gold collection constraints (never exceeds available gold)
- Performance metrics (cost, improvement, speedup)

View detailed results:
- JSON outputs: `outputs/alg_comparison_results.json`, `outputs/conservative_comparison_results.json`
- Markdown reports: Generated via `src/generate_report.py`

## Requirements

- `networkx >= 2.5` (graph operations)
- `numpy >= 1.19.0` (numerical computations)
- `matplotlib` (optional, for problem visualization)
- `icecream` (optional, for debugging)

Install dependencies:

```bash
pip install -r base_requirements.txt
# Or for full dependencies:
pip install -r requirements.txt
```

## Disclaimer

This project was developed with the assistance of AI tools, including code generation, optimization suggestions, and documentation support. The core algorithmic ideas and conceptual approach were developed through iterative experimentation and analysis of the problem structure. While AI assistance was used throughout the development process, all code has been reviewed, tested, and validated for correctness and performance by the author. The implementation logic, design decisions, and final solution represent the author's work, informed by algorithmic principles and empirical testing.
