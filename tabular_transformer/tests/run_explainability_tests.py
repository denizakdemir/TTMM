#!/usr/bin/env python
"""
Test runner for explainability tests.

This script provides a command-line interface for running the
explainability tests for different task heads.
"""

import os
import sys
import argparse
import pytest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run explainability tests for tabular transformer.")
    
    parser.add_argument(
        "--test-type",
        choices=["global", "local", "visualization", "sensitivity", "report", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--task-head",
        choices=["regression", "classification", "survival", "count", "competing_risks", "clustering", "all"],
        default="all",
        help="Task head to test (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--xvs",
        action="store_true",
        help="Show extra verbose and summary test output"
    )
    
    return parser.parse_args()


def get_test_pattern(test_type, task_head):
    """
    Generate the pytest test pattern based on test type and task head.
    
    Args:
        test_type: Type of tests to run
        task_head: Task head to test
        
    Returns:
        Test pattern string
    """
    # Base pattern
    pattern = "tabular_transformer/tests/explainability/"
    
    # Add test type filter
    if test_type != "all":
        pattern += f"test_{test_type}.py"
    else:
        pattern += "test_*.py"
    
    # Add task head filter
    if task_head != "all":
        pattern += f"::{task_head}"
    
    return pattern


def main():
    """Run the tests."""
    args = parse_args()
    
    # Change to project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Build test pattern
    test_pattern = get_test_pattern(args.test_type, args.task_head)
    
    # Build pytest arguments
    pytest_args = [test_pattern]
    
    # Add verbosity flags
    if args.verbose:
        pytest_args.append("-v")
    if args.xvs:
        pytest_args.extend(["-v", "-s"])
    
    # Print test information
    print(f"Running explainability tests with pattern: {test_pattern}")
    print(f"Pytest arguments: {pytest_args}")
    print("=" * 80)
    
    # Run the tests
    result = pytest.main(pytest_args)
    
    # Exit with appropriate code
    sys.exit(result)


if __name__ == "__main__":
    main()
