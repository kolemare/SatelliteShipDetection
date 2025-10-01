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

# Core dependencies for ConvNeXt training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pillow
pip install tqdm
pip install invoke

# Optional: plotting & JSON logging improvements
pip install matplotlib

echo "✅ All done. To activate your environment later, run:"
echo "source venv/bin/activate"
