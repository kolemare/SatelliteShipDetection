@echo off
setlocal enabledelayedexpansion

REM ── CONFIG ───────────────────────────────────────────────────────────
REM Common CUDA tags: cu118, cu121, cu124, cu126, cu128
if "%CUDA_WHL_TAG%"=="" set "CUDA_WHL_TAG=cu128"
if "%PYTORCH_VER%"==""   set "PYTORCH_VER=2.7.0"
REM Torch 2.7.0 pairs with torchvision 0.22.0 and torchaudio 2.7.0
if "%TORCHVISION_VER%"=="" set "TORCHVISION_VER=0.22.0"
if "%TORCHAUDIO_VER%"=="" set "TORCHAUDIO_VER=2.7.0"
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
pip install torch==%PYTORCH_VER% torchvision==%TORCHVISION_VER% torchaudio==%TORCHAUDIO_VER% --index-url https://download.pytorch.org/whl/%CUDA_WHL_TAG%
if errorlevel 1 (
    echo ⚠️  GPU wheel install failed. Falling back to CPU wheel…
    pip install torch==%PYTORCH_VER% torchvision==%TORCHVISION_VER% torchaudio==%TORCHAUDIO_VER% --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ❌ PyTorch install failed (GPU and CPU). Check internet or versions.
        exit /b 1
    )
)

echo 🧰 Installing project dependencies...
REM Core training + utilities
pip install pillow tqdm matplotlib invoke

REM Streamlit GUI + map stack
pip install streamlit streamlit-folium folium mercantile requests

REM Optional geospatial stack (uncomment as needed)
REM pip install shapely pyproj
REM pip install rasterio               ^^^ large wheel; needs matching GDAL
REM pip install pystac-client rio-tiler

echo 🔎 Verifying environment...
python -c "import platform,sys; print(f'Python : {platform.python_version()} ({sys.executable})'); \
import importlib, torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA   : {torch.cuda.is_available()} (device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})'); \
import streamlit, folium, mercantile, requests, PIL; print('Streamlit/Folium stack: OK')"

echo ✅ venv ready for PyTorch (%PYTORCH_VER%, %CUDA_WHL_TAG%).
echo 👉 Activate later with: venv\Scripts\activate
endlocal
