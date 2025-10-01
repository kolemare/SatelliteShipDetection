#!/bin/bash

# Exit immediately on error
set -e

# Create virtual environment in ./venv
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "✅ Virtual environment created in 'venv/'"
echo "🔁 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install dependencies
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install invoke

echo "✅ All done. To activate your environment, run:"
echo "source venv/bin/activate"
