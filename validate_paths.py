#!/usr/bin/env python3
"""
Validate solution paths using the correct is_valid function.
Checks that every consecutive pair of cities in the path has a direct edge.
"""

# Suppress matplotlib/font warnings when importing Problem (uses matplotlib)
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import importlib.util
import re
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Problem import Problem
from src.generate_test_cases import test_cases


def is_valid(problem, path):
    """Check that each consecutive pair of cities has a direct edge in the graph."""
    for (c1, gold1), (c2, gold2) in zip(path, path[1:]):
        yield problem.graph.has_edge(c1, c2)


def validate_single_path(problem, path):
    """Return True iff all consecutive edges exist in the graph."""
    return all(is_valid(problem, path))


def discover_student_solutions():
    """Find all s<digits>.py files in project root."""
    pattern = re.compile(r"^s\d+\.py$")
    found = []
    for f in PROJECT_ROOT.iterdir():
        if f.is_file() and pattern.match(f.name):
            found.append(f)
    return sorted(found, key=lambda p: p.name)


def run_validation(use_full_test_suite=True):
    """Run validation on all student solutions."""
    # Use subset for quick check, or full suite
    if use_full_test_suite:
        cases = test_cases
    else:
        # Quick check: first 10 diverse cases
        cases = test_cases[:10]

    student_files = discover_student_solutions()
    if not student_files:
        print("No student solution files (s*.py) found in project root.")
        return

    results = []  # (student, test_id, valid, failed_edge or None)
    errors = []   # (student, test_id, exception)

    for student_file in student_files:
        module_name = student_file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, student_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            solution_fn = getattr(module, "solution", None)
            if solution_fn is None:
                print(f"  Warning: {module_name} has no solution() function, skipping.")
                continue
        except Exception as e:
            print(f"  Error loading {module_name}: {e}")
            errors.append((module_name, None, str(e)))
            continue

        for test_config in cases:
            test_id = test_config.get("id", "?")
            try:
                problem = Problem(
                    num_cities=test_config["num_cities"],
                    alpha=test_config["alpha"],
                    beta=test_config["beta"],
                    density=test_config["density"],
                    seed=test_config["seed"],
                )
                path = solution_fn(problem)
            except Exception as e:
                errors.append((module_name, test_id, str(e)))
                results.append((module_name, test_id, False, None))
                continue

            if not path or len(path) < 2:
                results.append((module_name, test_id, False, "empty or trivial path"))
                continue

            # Find first failed edge if any
            failed_edge = None
            for (c1, gold1), (c2, gold2) in zip(path, path[1:]):
                if not problem.graph.has_edge(c1, c2):
                    failed_edge = (c1, c2)
                    break

            valid = failed_edge is None
            results.append((module_name, test_id, valid, failed_edge))

    # Report
    print("=" * 60)
    print("PATH VALIDITY VALIDATION")
    print("=" * 60)
    print(f"Student files: {[f.name for f in student_files]}")
    print(f"Test cases: {len(cases)}")
    print()

    valid_count = sum(1 for _, _, v, _ in results if v)
    total_count = len(results)
    invalid_count = total_count - valid_count

    if invalid_count == 0 and not errors:
        print("RESULT: All paths are VALID.")
    else:
        print(f"RESULT: {invalid_count} of {total_count} paths are INVALID.")
        if errors:
            print(f"         {len(errors)} run/import error(s).")
        print()

        # Details for invalid paths
        invalid_results = [(s, t, e) for s, t, v, e in results if not v]
        if invalid_results:
            print("Invalid paths:")
            for student, test_id, failed_edge in invalid_results[:20]:
                edge_str = f"missing edge {failed_edge}" if failed_edge else "error"
                print(f"  - {student} test {test_id}: {edge_str}")
            if len(invalid_results) > 20:
                print(f"  ... and {len(invalid_results) - 20} more")

        if errors:
            print()
            print("Errors:")
            for student, test_id, msg in errors[:10]:
                tid = f" test {test_id}" if test_id else ""
                print(f"  - {student}{tid}: {msg}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

    print()
    print("=" * 60)
    return valid_count == total_count and not errors


if __name__ == "__main__":
    full_suite = "--quick" not in sys.argv
    if full_suite:
        print("Using full test suite (43 cases). Use --quick for 10 cases.\n")
    else:
        print("Using quick test suite (10 cases).\n")
    run_validation(use_full_test_suite=full_suite)
