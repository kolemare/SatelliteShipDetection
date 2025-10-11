@echo off
setlocal enabledelayedexpansion

rem --- Config (override by pre-setting env vars) ---
if not defined CUDA_WHL_TAG set "CUDA_WHL_TAG=cu128"
if not defined PYTORCH_VER  set "PYTORCH_VER=2.7.0"
if not defined TORCHVISION_VER set "TORCHVISION_VER=0.22.0"
if not defined TORCHAUDIO_VER set "TORCHAUDIO_VER=2.7.0"
if not defined VENV_DIR set "VENV_DIR=venv"

echo 🔧 Creating venv: %VENV_DIR%
python -m venv "%VENV_DIR%"
if errorlevel 1 goto :fail

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :fail

python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo 🧠 Installing PyTorch (%CUDA_WHL_TAG%)...
pip install ^
  torch==%PYTORCH_VER% ^
  torchvision==%TORCHVISION_VER% ^
  torchaudio==%TORCHAUDIO_VER% ^
  --index-url https://download.pytorch.org/whl/%CUDA_WHL_TAG%
if errorlevel 1 (
  echo ⚠️  GPU wheel failed, falling back to CPU wheels…
  pip install ^
    torch==%PYTORCH_VER% ^
    torchvision==%TORCHVISION_VER% ^
    torchaudio==%TORCHAUDIO_VER% ^
    --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto :fail
)

echo 📦 Installing project deps…
pip install pillow tqdm matplotlib invoke
if errorlevel 1 goto :fail

pip install streamlit streamlit-folium folium mercantile requests
if errorlevel 1 goto :fail

rem Real-ESRGAN + basicsr + OpenCV (CPU/CUDA-wheels resolved by pip)
pip install realesrgan basicsr opencv-python
if errorlevel 1 goto :fail

echo ✅ venv ready. Activate with: call "%VENV_DIR%\Scripts\activate.bat"
goto :eof

:fail
echo ❌ Setup failed. Check the messages above.
exit /b 1
