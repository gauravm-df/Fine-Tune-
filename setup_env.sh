#!/bin/bash
set -e

echo "=========================================================="
echo "Setting up Virtual Environment for Existing LLM Project..."
echo "=========================================================="

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup Complete! Activate your environment with:"
echo "source venv/bin/activate"

