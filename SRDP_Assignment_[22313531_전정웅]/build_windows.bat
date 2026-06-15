@echo off
python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    python -m pip install -r requirements.txt
    python -m PyInstaller --noconfirm --onefile --windowed --name SRDP_Assignment_GUI src\srdp_gui.py
    exit /b %ERRORLEVEL%
)

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3 -m pip install -r requirements.txt
    py -3 -m PyInstaller --noconfirm --onefile --windowed --name SRDP_Assignment_GUI src\srdp_gui.py
    exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Install Python 3 first.
exit /b 1
