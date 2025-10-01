@echo off
echo 🔧 Creating virtual environment...

python -m venv venv

if errorlevel 1 (
    echo ❌ Failed to create virtual environment.
    exit /b 1
)

echo ✅ Virtual environment created in 'venv\'
echo 🔁 Activating virtual environment...

call venv\Scripts\activate.bat

echo 📦 Installing required packages...
pip install --upgrade pip

:: Core dependencies for ConvNeXt training (CPU build of PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pillow
pip install tqdm
pip install invoke

:: Optional: plotting and analysis
pip install matplotlib

echo ✅ All done.
echo To activate your virtual environment later, run:
echo venv\Scripts\activate
