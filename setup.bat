@echo off
REM One-time setup on a new computer: installs the Python libraries into a local .venv folder.
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install, then run setup.bat again.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist cfbd_key.txt (
  > cfbd_key.txt echo # Paste your CollegeFootballData.com API key on the line below, then save.
  echo Created cfbd_key.txt - paste your CollegeFootballData.com key into it for college picks.
)
echo.
echo Setup finished. Double-click run_weekly.bat whenever you want new picks.
pause
