#!/usr/bin/env bash
set -euo pipefail

# Config (override via env): CUDA_WHL_TAG=cu121 PYTORCH_VER=2.7.0 TORCHVISION_VER=0.22.0 TORCHAUDIO_VER=2.7.0 VENV_DIR=venv
CUDA_WHL_TAG="${CUDA_WHL_TAG:-cu128}"
PYTORCH_VER="${PYTORCH_VER:-2.7.0}"
TORCHVISION_VER="${TORCHVISION_VER:-0.22.0}"
TORCHAUDIO_VER="${TORCHAUDIO_VER:-2.7.0}"
VENV_DIR="${VENV_DIR:-venv}"

echo "🔧 Creating venv: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "🧠 Installing PyTorch (${CUDA_WHL_TAG})..."
set +e
pip install \
  "torch==${PYTORCH_VER}" \
  "torchvision==${TORCHVISION_VER}" \
  "torchaudio==${TORCHAUDIO_VER}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "⚠️  GPU wheel failed, falling back to CPU wheels…"
  pip install \
    "torch==${PYTORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    "torchaudio==${TORCHAUDIO_VER}" \
    --index-url "https://download.pytorch.org/whl/cpu"
fi

echo "📦 Installing project deps…"
pip install pillow tqdm matplotlib invoke
pip install streamlit streamlit-folium folium mercantile requests
# Real-ESRGAN (CPU/CUDA) + basicsr + OpenCV
pip install realesrgan basicsr opencv-python

echo "✅ venv ready. Activate with: source ${VENV_DIR}/bin/activate"
