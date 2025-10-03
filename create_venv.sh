#!/usr/bin/env bash
set -euo pipefail

# --- Choose your CUDA wheel tag and PyTorch version ---
# Common tags: cu118, cu121, cu124, cu126, cu128
CUDA_WHL_TAG="cu128"
PYTORCH_VER="2.7.0"

echo "🔧 Creating virtual environment..."
python3 -m venv venv

echo "🔁 Activating virtual environment..."
source venv/bin/activate

echo "📦 Upgrading pip & wheel..."
python -m pip install --upgrade pip wheel

echo "🧠 Installing GPU-enabled PyTorch ${PYTORCH_VER} (${CUDA_WHL_TAG})..."
pip install "torch==${PYTORCH_VER}" torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"

echo "🧰 Installing project deps..."
pip install pillow tqdm invoke matplotlib

echo "✅ venv ready for PyTorch (${PYTORCH_VER}, ${CUDA_WHL_TAG})."
echo "👉 Activate: source venv/bin/activate"
python - <<'PY'
import torch, platform
print(f"Python : {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()} (device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
PY
