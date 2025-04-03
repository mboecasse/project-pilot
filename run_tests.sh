#!/bin/bash

# Run pytest with coverage
pytest --cov=app --cov-report=term --cov-report=html tests/

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed."
    exit 1
fi