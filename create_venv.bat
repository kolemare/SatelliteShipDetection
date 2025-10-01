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
pip install invoke

echo ✅ All done.
echo To activate your virtual environment later, run:
echo venv\Scripts\activate
