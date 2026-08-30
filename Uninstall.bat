@echo off
title SG CUBE Uninstaller
echo ============================================================
echo   SG CUBE Uninstaller
echo ============================================================
echo.
echo Closing all running SG CUBE processes...
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo Removing Windows Startup shortcut...
del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SG CUBE Wake Listener.lnk" >nul 2>&1

echo Removing Registry auto-start entry...
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "SGCubeWakeListener" /f >nul 2>&1

echo Removing Start Menu and Desktop shortcuts...
del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\SG CUBE.lnk" >nul 2>&1
del /f /q "%USERPROFILE%\Desktop\SG CUBE.lnk" >nul 2>&1

echo SG CUBE has been successfully removed.
