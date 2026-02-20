@echo off
echo ========================================
echo  Captcha CLIP Solver  ^|  FastAPI Server
echo ========================================

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    pause & exit /b 1
)

:: Create .env from example if missing
if not exist .env (
    copy .env.example .env
    echo [INFO] Created .env from .env.example
    echo [ACTION] Please open .env and set your API_TOKEN before continuing.
    pause
)

:: Create and activate venv
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

:: Install dependencies
echo [INFO] Installing dependencies (this may take a while for torch + clip)...
pip install -r requirements.txt

:: Start server
echo.
echo [INFO] Starting server on http://localhost:8000
echo [INFO] Docs available at http://localhost:8000/docs
echo.
python main.py
