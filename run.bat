@echo off
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cargo run -- %*
if errorlevel 1 pause
