@echo off
rem Wrapper so the build can run on machines where the PowerShell execution policy
rem forbids .ps1 files (the default on this workstation).
rem
rem Usage:
rem   scripts\build_submission.cmd
rem   scripts\build_submission.cmd -Smoke
rem   scripts\build_submission.cmd -SkipSave -Tag pcb-parser:dev
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_submission.ps1" %*
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" echo.
if not "%CODE%"=="0" echo build_submission failed with exit code %CODE%
exit /b %CODE%
