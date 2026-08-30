@echo off
title SG CUBE 2.4.6 Windows Installer
setlocal enabledelayedexpansion

echo ============================================================
echo        SG CUBE 2.4.6 — OFFICIAL WINDOWS INSTALLER          
echo ============================================================
echo.

set "INSTALL_DIR=%LOCALAPPDATA%\Programs\SG-CUBE"
echo [1/5] Target Installation Directory: !INSTALL_DIR!

echo [2/5] Terminating any existing SG CUBE processes...
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo [3/5] Deploying SG CUBE application files and runtime...
if not exist "!INSTALL_DIR!" mkdir "!INSTALL_DIR!"
xcopy "%~dp0*" "!INSTALL_DIR!\" /E /I /Y /Q >nul

echo [4/5] Creating Windows Start Menu and Desktop Shortcuts...
set "VBS_SCRIPT=%TEMP%\CreateSGCubeShortcuts.vbs"
(
    echo Set oWS = WScript.CreateObject("WScript.Shell"^)
    echo sLinkFile = oWS.ExpandEnvironmentStrings("%APPDATA%\Microsoft\Windows\Start Menu\Programs\SG CUBE.lnk"^)
    echo Set oLink = oWS.CreateShortcut(sLinkFile^)
    echo oLink.TargetPath = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\run.bat"^)
    echo oLink.WorkingDirectory = "!INSTALL_DIR!"
    echo oLink.IconLocation = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\assets\SG-CUBE.ico,0"^)
    echo oLink.Description = "SG CUBE — Multimodal AI Assistant"
    echo oLink.Save
    
    echo sDeskFile = oWS.ExpandEnvironmentStrings("%USERPROFILE%\Desktop\SG CUBE.lnk"^)
    echo Set oDesk = oWS.CreateShortcut(sDeskFile^)
    echo oDesk.TargetPath = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\run.bat"^)
    echo oDesk.WorkingDirectory = "!INSTALL_DIR!"
    echo oDesk.IconLocation = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\assets\SG-CUBE.ico,0"^)
    echo oDesk.Description = "SG CUBE — Multimodal AI Assistant"
    echo oDesk.Save

    echo sStartupFile = oWS.ExpandEnvironmentStrings("%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SG CUBE Wake Listener.lnk"^)
    echo Set oStartup = oWS.CreateShortcut(sStartupFile^)
    echo oStartup.TargetPath = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\run_wake_listener.bat"^)
    echo oStartup.WorkingDirectory = "!INSTALL_DIR!"
    echo oStartup.IconLocation = oWS.ExpandEnvironmentStrings("!INSTALL_DIR!\assets\SG-CUBE.ico,0"^)
    echo oStartup.WindowStyle = 7
    echo oStartup.Description = "SG CUBE Background Wake Listener"
    echo oStartup.Save
) > "%VBS_SCRIPT%"
cscript //nologo "%VBS_SCRIPT%"
del "%VBS_SCRIPT%" >nul 2>&1

echo [5/5] Configuring Windows Startup Registry Auto-Start Entry...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "SGCubeWakeListener" /t REG_SZ /d "\"!INSTALL_DIR!\run_wake_listener.bat\"" /f >nul 2>&1

echo.
echo ============================================================
echo    SUCCESS: SG CUBE 2.4.6 HAS BEEN SUCCESSFULLY INSTALLED!  
echo ============================================================
echo   Location: !INSTALL_DIR!
echo   Launch from: Desktop Shortcut, Start Menu, or say "Hey SG CUBE"
echo ============================================================
