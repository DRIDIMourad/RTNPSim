@echo off
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%
cd /d "%SCRIPT_DIR%"
python app_tkinter.py
pause
