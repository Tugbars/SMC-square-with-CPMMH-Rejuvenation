@echo off
REM ============================================================================
REM Build script for SMC² RBPF with Intel ICX on Windows
REM ============================================================================
REM 
REM Prerequisites:
REM   1. Intel oneAPI Base Toolkit installed (includes ICX compiler)
REM   2. CMake 3.20+ in PATH
REM   3. Ninja (recommended) or Visual Studio 2019/2022
REM
REM Usage:
REM   build.bat [Release|Debug|RelWithDebInfo]
REM ============================================================================

setlocal enabledelayedexpansion

REM Default build type
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo.
echo ============================================
echo  SMC2 RBPF Build Script (Intel ICX)
echo ============================================
echo  Build Type: %BUILD_TYPE%
echo.

REM ============================================================================
REM Initialize Intel oneAPI environment
REM ============================================================================

REM Try common installation paths
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if not exist "%ONEAPI_ROOT%" (
    set "ONEAPI_ROOT=C:\Program Files\Intel\oneAPI"
)

if exist "%ONEAPI_ROOT%\setvars.bat" (
    echo Initializing Intel oneAPI environment...
    call "%ONEAPI_ROOT%\setvars.bat" intel64 vs2022 >nul 2>&1
    if errorlevel 1 (
        call "%ONEAPI_ROOT%\setvars.bat" intel64 vs2019 >nul 2>&1
    )
) else (
    echo WARNING: Intel oneAPI setvars.bat not found at %ONEAPI_ROOT%
    echo Please ensure Intel oneAPI is installed and ICX is in PATH
    echo.
)

REM Verify ICX is available
where icx >nul 2>&1
if errorlevel 1 (
    echo ERROR: Intel ICX compiler not found in PATH
    echo Please install Intel oneAPI Base Toolkit or run setvars.bat manually
    exit /b 1
)

echo Using compiler: 
icx --version 2>nul | findstr /i "Intel"
echo.

REM ============================================================================
REM Create build directory and run CMake
REM ============================================================================

set BUILD_DIR=build_%BUILD_TYPE%

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

cd %BUILD_DIR%

REM Prefer Ninja if available, fallback to NMake
where ninja >nul 2>&1
if errorlevel 1 (
    echo Using NMake generator...
    set CMAKE_GENERATOR=NMake Makefiles
) else (
    echo Using Ninja generator...
    set CMAKE_GENERATOR=Ninja
)

echo.
echo Running CMake configuration...
cmake -G "%CMAKE_GENERATOR%" ^
    -DCMAKE_C_COMPILER=icx ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    ..

if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

echo.
echo Building...
cmake --build . --config %BUILD_TYPE% -j

if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)

cd ..

echo.
echo ============================================
echo  Build successful!
echo  Executable: %BUILD_DIR%\bin\test_smc2.exe
echo ============================================
echo.

REM Run test if requested
if "%2"=="--test" (
    echo Running tests...
    %BUILD_DIR%\bin\test_smc2.exe
)

endlocal
