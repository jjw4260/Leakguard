@echo off
python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    python src\srdp_full_safe_framework.py --epochs 6 --trials 10 --out-prefix outputs\srdp_assignment
    exit /b %ERRORLEVEL%
)

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3 src\srdp_full_safe_framework.py --epochs 6 --trials 10 --out-prefix outputs\srdp_assignment
    exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Install Python 3 and run pip install -r requirements.txt.
exit /b 1
