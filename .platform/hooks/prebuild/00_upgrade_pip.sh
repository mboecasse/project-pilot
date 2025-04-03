#!/usr/bin/env bash

echo "Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel

echo "Pip version:"
python3 -m pip --version

echo "Setuptools version:"
python3 -c "import setuptools; print(f'setuptools {setuptools.__version__}')"

echo "Wheel version:"
python3 -c "import wheel; print(f'wheel {wheel.__version__}')"

echo "Pip upgrade completed."