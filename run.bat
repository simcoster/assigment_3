@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="cli" goto cli
if /i "%~1"=="streamlit" goto streamlit

echo Usage: %~nx0 cli [main.py options...] ^| streamlit
echo.
echo   cli         Terminal agent  (uv run python main.py)
echo   streamlit   Web chat UI     (uv run streamlit run streamlit_app.py)
exit /b 1

:cli
uv run python main.py %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:streamlit
uv run streamlit run streamlit_app.py
exit /b %ERRORLEVEL%
