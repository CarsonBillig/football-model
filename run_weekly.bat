@echo off
REM Weekly run: grade last week, retrain, predict this week (NFL + college), rebuild docs\index.html,
REM open it, and publish it to GitHub if this folder is connected to a GitHub repository.
REM Extra options pass straight through, e.g.  run_weekly.bat --backtest   or   run_weekly.bat --nfl-only
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe update_model.py %*
if errorlevel 1 (
  echo.
  echo The update failed - read the messages above.
  pause
  exit /b 1
)
start "" "docs\index.html"
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo Not connected to GitHub yet - the site was only updated on this computer.
) else (
  echo Publishing to GitHub...
  git add -A
  git commit -q -m "Weekly update %date% %time:~0,5%"
  git push -q
  if errorlevel 1 (echo GitHub push failed - check your internet connection or GitHub sign-in.) else (echo Published.)
)
pause
