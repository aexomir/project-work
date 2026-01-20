"""
Run baseline algorithm - reproduces Cell 5 from notebooks/problem.ipynb
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.baseline_impl import Problem
from icecream import ic


def main():
    logging.getLogger().setLevel(logging.WARNING)

    results = []
    
    result = ic(Problem(100, density=0.2, alpha=1, beta=1).baseline())
    results.append(("Problem(100, density=0.2, alpha=1, beta=1)", result))
    
    result = ic(Problem(100, density=0.2, alpha=2, beta=1).baseline())
    results.append(("Problem(100, density=0.2, alpha=2, beta=1)", result))
    
    result = ic(Problem(100, density=0.2, alpha=1, beta=2).baseline())
    results.append(("Problem(100, density=0.2, alpha=1, beta=2)", result))
    
    result = ic(Problem(100, density=1, alpha=1, beta=1).baseline())
    results.append(("Problem(100, density=1, alpha=1, beta=1)", result))
    
    result = ic(Problem(100, density=1, alpha=2, beta=1).baseline())
    results.append(("Problem(100, density=1, alpha=2, beta=1)", result))
    
    result = ic(Problem(100, density=1, alpha=1, beta=2).baseline())
    results.append(("Problem(100, density=1, alpha=1, beta=2)", result))
    
    result = ic(Problem(1_000, density=0.2, alpha=1, beta=1).baseline())
    results.append(("Problem(1_000, density=0.2, alpha=1, beta=1)", result))
    
    result = ic(Problem(1_000, density=0.2, alpha=2, beta=1).baseline())
    results.append(("Problem(1_000, density=0.2, alpha=2, beta=1)", result))
    
    result = ic(Problem(1_000, density=0.2, alpha=1, beta=2).baseline())
    results.append(("Problem(1_000, density=0.2, alpha=1, beta=2)", result))
    
    result = ic(Problem(1_000, density=1, alpha=1, beta=1).baseline())
    results.append(("Problem(1_000, density=1, alpha=1, beta=1)", result))
    
    result = ic(Problem(1_000, density=1, alpha=2, beta=1).baseline())
    results.append(("Problem(1_000, density=1, alpha=2, beta=1)", result))
    
    result = ic(Problem(1_000, density=1, alpha=1, beta=2).baseline())
    results.append(("Problem(1_000, density=1, alpha=1, beta=2)", result))

    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "baseline_results.txt"
    with open(output_file, 'w') as f:
        f.write("Baseline Results\n")
        f.write("=" * 80 + "\n\n")
        for params, result in results:
            f.write(f"{params}: {result}\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

