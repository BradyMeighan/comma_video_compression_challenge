@echo off
REM Launcher for grid_search.py with correct Python venv and ffmpeg
REM Usage: run_grid_search.bat [args...]
REM Example: run_grid_search.bat --device cuda --quick --codec svtav1

set "VENV=%~dp0.venv\Scripts"
set "FFMPEG=C:\Users\modce\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
set "PATH=%VENV%;%FFMPEG%;%PATH%"
set "PYTHONUTF8=1"

echo Python: & python --version
echo FFmpeg: & ffmpeg -version 2>&1 | findstr "ffmpeg version"
echo.

python "%~dp0grid_search.py" %*
