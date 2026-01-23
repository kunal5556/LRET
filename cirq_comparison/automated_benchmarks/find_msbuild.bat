@echo off
echo Searching for MSBuild on this system...
echo.

echo Checking Program Files locations...
for /f "delims=" %%i in ('dir "C:\Program Files\Microsoft Visual Studio" /s /b /a:d 2^>nul ^| findstr /i "MSBuild\\.*\\Bin$"') do (
    if exist "%%i\MSBuild.exe" (
        echo FOUND: "%%i\MSBuild.exe"
    )
)

echo.
echo Checking Program Files (x86) locations...
for /f "delims=" %%i in ('dir "C:\Program Files (x86)\Microsoft Visual Studio" /s /b /a:d 2^>nul ^| findstr /i "MSBuild\\.*\\Bin$"') do (
    if exist "%%i\MSBuild.exe" (
        echo FOUND: "%%i\MSBuild.exe"
    )
)

echo.
echo Checking if MSBuild is in PATH...
where msbuild 2>nul
if errorlevel 1 (
    echo MSBuild NOT in PATH
) else (
    echo MSBuild found in PATH
)

echo.
echo Checking VS Developer Command Prompt...
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    echo VS 2022 Community Developer Command Prompt: FOUND
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" (
    echo VS 2022 Professional Developer Command Prompt: FOUND
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" (
    echo VS 2022 Enterprise Developer Command Prompt: FOUND
)

echo.
echo ========================================================================
echo Please copy the FOUND paths above and share them
echo ========================================================================
pause
