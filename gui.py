#!/usr/bin/env python3
"""
Launcher for the Streamlit GUI that lives in gui/gui_app.py.

Usage:
  python gui.py
"""
import os
import sys
import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.resolve()
    app_file = repo_root / "application" / "gui_app.py"
    
    if not app_file.exists():
        print(f"❌ GUI app not found at: {app_file}")
        sys.exit(1)

    # Forward any additional CLI args to streamlit if provided
    extra = sys.argv[1:]

    # Run: streamlit run gui/gui_app.py [extra...]
    cmd = ["streamlit", "run", str(app_file), "--server.headless=false", *extra]
    # If streamlit not on PATH, fallback to python -m streamlit
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.headless=true", *extra]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
