@echo off
setlocal enabledelayedexpansion

REM ── CONFIG ───────────────────────────────────────────────────────────
REM Common CUDA tags: cu118, cu121, cu124, cu126, cu128
set "CUDA_WHL_TAG=cu128"
set "PYTORCH_VER=2.7.0"
REM Torch 2.7.0 pairs with torchvision 0.22.0 and torchaudio 2.7.0
set "TORCHVISION_VER=0.22.0"
set "TORCHAUDIO_VER=2.7.0"
REM ─────────────────────────────────────────────────────────────────────

echo 🔧 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment.
    exit /b 1
)

echo 🔁 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📦 Upgrading build tools...
python -m pip install --upgrade pip setuptools wheel

echo 🧠 Installing GPU-enabled PyTorch %PYTORCH_VER% (%CUDA_WHL_TAG%)...
pip install torch==%PYTORCH_VER% torchvision==%TORCHVISION_VER% torchaudio==%TORCHAUDIO_VER% ^
  --index-url https://download.pytorch.org/whl/%CUDA_WHL_TAG%

if errorlevel 1 (
    echo ❌ PyTorch install failed. Check your CUDA tag or internet connection.
    exit /b 1
)

echo 🧰 Installing project dependencies...
REM Core training + utilities
pip install pillow tqdm matplotlib invoke

REM GUI: PyQt + WebEngine + map tiling helpers
pip install PyQt6 PyQt6-WebEngine mercantile requests

REM Optional geospatial stack (uncomment as needed)
REM pip install shapely pyproj
REM pip install rasterio              ^^^ note: large wheel; needs GDAL wheels
REM pip install pystac-client rio-tiler

echo 🔎 Verifying environment...
python - <<PY
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

echo ✅ venv ready for PyTorch (%PYTORCH_VER%, %CUDA_WHL_TAG%).
echo 👉 Activate later with: venv\Scripts\activate
endlocal
