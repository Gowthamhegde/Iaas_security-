@echo off
REM ── IaaS Security ML-IDS — Windows Quick Start ────────────────────────────
REM Run this script once from the project folder.

cd /d "%~dp0"
echo.
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/3] Quick-training the Random Forest model (no grid search)...
python quick_train.py
if errorlevel 1 (
    echo ERROR: Training failed. See output above.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching Streamlit dashboard...
echo       Open http://localhost:8501 in your browser
echo       Press Ctrl+C to stop.
echo.
streamlit run dashboard\app.py --server.port 8501 --theme.base dark
pause
