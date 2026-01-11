@echo off
title LRET Benchmark - 4q 25s 100e 10n (Fast)
cd /d D:\LRET
echo ============================================================
echo LRET vs default.mixed Benchmark (Fast Version)
echo ============================================================
echo Config: 4 qubits, 25 samples, 100 epochs, 10%% noise
echo Estimated: LRET ~2.7h, default.mixed ~26h
echo Started: %date% %time%
echo ============================================================
echo.

python benchmarks\benchmark_4q_25s_100e_10n.py

echo.
echo ============================================================
echo Benchmark finished at: %date% %time%
echo ============================================================
pause
