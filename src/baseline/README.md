# Baseline Implementation

This directory contains the baseline algorithm extracted from `notebooks/problem.ipynb`.

## Files

- **`baseline_impl.py`**: Core implementation of the `Problem` class and baseline algorithm
  - Exact copy of the algorithm from the notebook
  - No logic changes, only minimal formatting for module structure
  
- **`run_baseline.py`**: Command-line runner that reproduces Cell 5 from the notebook
  - Executes all 12 test cases with identical parameters
  - Saves results to `outputs/baseline_results.txt`

- **`__init__.py`**: Package initialization file

## Usage

Run the baseline from the command line:

```bash
python src/baseline/run_baseline.py
```

This will:
1. Execute the baseline algorithm with the same 12 parameter configurations as the notebook
2. Print results to stdout (using icecream)
3. Save results to `outputs/baseline_results.txt`

## Results Verification

The command-line execution produces **identical results** to running the notebook top-to-bottom:
- Same random seed (42) used by default
- Same algorithm logic
- Same parameter values
- Results match within floating-point precision

## Implementation Notes

**Changes from notebook:**
- Added module docstrings
- Removed plot execution (Cell 4) - plot method still available but not called
- Added output file saving functionality

**No changes to:**
- Algorithm logic or control flow
- Mathematical operations
- Variable names
- Default parameter values
- Random number generation

