@echo off
echo ============================================================
echo Creating a clean Python Virtual Environment...
echo ============================================================

:: Check if venv already exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Initializing virtual environment in .\venv ...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

echo.
echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat

:: Upgrade pip and install requirements (with no-cache to ensure clean binaries)
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

echo.
echo ============================================================
echo [3/3] Running Training Pipeline...
echo ============================================================
python pipeline.py --train

echo.
echo Pipeline finished.
pause
