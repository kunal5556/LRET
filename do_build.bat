@echo off
cd /d D:\LRET\build
echo BUILD_START=%date% %time% > D:\LRET\build_done.txt
cmake --build . --config Release >> D:\LRET\build_done.txt 2>&1
echo BUILD_EXIT=%ERRORLEVEL% >> D:\LRET\build_done.txt
echo BUILD_END=%date% %time% >> D:\LRET\build_done.txt
if exist Release\test_mpi_scatter.exe (
    echo TEST_EXE=EXISTS >> D:\LRET\build_done.txt
) else (
    echo TEST_EXE=MISSING >> D:\LRET\build_done.txt
)
