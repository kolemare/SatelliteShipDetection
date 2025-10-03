#!/usr/bin/env bash
set -euo pipefail

# ── CONFIG ─────────────────────────────────────────────────────────────
# Common CUDA tags: cu118, cu121, cu124, cu126, cu128
CUDA_WHL_TAG="cu128"
PYTORCH_VER="2.7.0"
# Torch 2.7.0 pairs with torchvision 0.22.0 and torchaudio 2.7.0
TORCHVISION_VER="0.22.0"
TORCHAUDIO_VER="2.7.0"
# ──────────────────────────────────────────────────────────────────────

echo "🔧 Creating virtual environment..."
python3 -m venv venv

echo "🔁 Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "📦 Upgrading build tools..."
python -m pip install --upgrade pip setuptools wheel

echo "🧠 Installing GPU-enabled PyTorch ${PYTORCH_VER} (${CUDA_WHL_TAG})..."
pip install "torch==${PYTORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"

echo "🧰 Installing project dependencies..."
# Core training + utilities
pip install pillow tqdm matplotlib invoke

# GUI: PyQt + WebEngine + map tiling helpers
pip install PyQt6 PyQt6-WebEngine mercantile requests

# Optional geospatial stack (uncomment as needed)
# pip install shapely pyproj
# pip install rasterio              # note: large wheel; needs manylibs wheel
# pip install pystac-client rio-tiler

echo "🔎 Verifying environment..."
python - <<'PY'
import platform, sys
print(f"Python : {platform.python_version()} ({sys.executable})")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA   : {torch.cuda.is_available()} (device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
except Exception as e:
    print("PyTorch import error:", e)

try:
    import PyQt6, PyQt6.QtWebEngineWidgets
    print("PyQt6  : OK (WebEngine available)")
except Exception as e:
    print("PyQt6/WebEngine import error:", e)
PY

echo "✅ venv ready for PyTorch (${PYTORCH_VER}, ${CUDA_WHL_TAG})."
echo "👉 Activate: source venv/bin/activate"
