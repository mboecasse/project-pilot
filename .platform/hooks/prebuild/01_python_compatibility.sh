#!/usr/bin/env bash

echo "Running Python 3.13 compatibility checks..."

# Ensure Python 3.13 is properly set up
echo "Python version:"
python --version

# Test Python 3.13 specific features
python -c "print('Testing Python 3.13 features: PEP 695 Type Aliases')"
python -c "type Point = tuple[float, float]; print('Type alias feature works')"

echo "Compatibility checks completed."