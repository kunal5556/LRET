# LRET Full Validation Benchmark Suite
# Phase A.2: Comprehensive Baseline vs Optimized Comparison
# 
# Usage: .\run_full_benchmark.ps1 [-Quick] [-OutputDir results]
#
# This script runs reproducible benchmarks with fixed seeds across:
# - Multiple qubit counts (8, 10, 11, 12)
# - Multiple initial ranks (4, 8, 16, 32)
# - All parallelization modes (sequential, row, column, batch, hybrid)
# - Multiple trials for statistical significance

param(
    [switch]$Quick,           # Quick mode: fewer configurations
    [int]$Trials = 3,         # Number of trials per configuration
    [string]$OutputDir = "results",
    [int]$Seed = 42           # Fixed seed for reproducibility
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Configuration
if ($Quick) {
    $QubitCounts = @(8, 10)
    $InitialRanks = @(8, 16)
    $Modes = @("row", "column")
    $Depth = 15
    $Trials = 2
} else {
    $QubitCounts = @(8, 10, 11)
    $InitialRanks = @(4, 8, 16, 32)
    $Modes = @("sequential", "row", "column", "batch", "hybrid")
    $Depth = 20
}

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║     LRET Validation Benchmark Suite - Phase A.2                ║"
Write-Host "╠════════════════════════════════════════════════════════════════╣"
Write-Host "║ Timestamp: $timestamp                                  ║"
Write-Host "║ Mode: $(if ($Quick) { 'QUICK' } else { 'FULL' })                                                 ║"
Write-Host "║ Trials: $Trials per configuration                               ║"
Write-Host "║ Seed: $Seed (fixed for reproducibility)                        ║"
Write-Host "║ Qubits: $($QubitCounts -join ', ')                                          ║"
Write-Host "║ Ranks: $($InitialRanks -join ', ')                                          ║"
Write-Host "║ Modes: $($Modes -join ', ')               ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Ensure output directory exists
$outPath = Join-Path $OutputDir $timestamp
New-Item -ItemType Directory -Path $outPath -Force | Out-Null
Write-Host "Output directory: $outPath"
Write-Host ""

# Results storage
$allResults = @()
$totalConfigs = $QubitCounts.Count * $InitialRanks.Count * $Modes.Count * 2  # x2 for baseline+optimized
$currentConfig = 0

foreach ($n in $QubitCounts) {
    foreach ($rank in $InitialRanks) {
        foreach ($mode in $Modes) {
            $currentConfig++
            $pct = [Math]::Round(($currentConfig / $totalConfigs) * 100)
            Write-Host "[$pct%] Config: n=$n, rank=$rank, mode=$mode" -ForegroundColor Cyan
            
            # Run baseline trials
            $baselineTimes = @()
            $baselineRanks = @()
            for ($t = 1; $t -le $Trials; $t++) {
                Write-Host "  Baseline trial $t/$Trials..." -NoNewline
                $trialSeed = $Seed + $t  # Vary seed per trial but reproducibly
                try {
                    $output = & .\baseline\quantum_sim.exe -n $n -d $Depth --initial-rank $rank --mode $mode --seed $trialSeed --allow-swap --non-interactive 2>&1
                    $outputStr = $output | Out-String
                    
                    # Extract timing from output
                    $timePattern = 'Time:\s+([0-9.]+)\s*s'
                    if ($outputStr -match $timePattern) {
                        $time = [double]$Matches[1]
                        $baselineTimes += $time
                        Write-Host " $([Math]::Round($time, 3))s" -ForegroundColor Green
                    } else {
                        Write-Host " (parse error)" -ForegroundColor Yellow
                    }
                    
                    # Extract final rank
                    $rankPattern = 'Final Rank:\s+([0-9]+)'
                    if ($outputStr -match $rankPattern) {
                        $baselineRanks += [int]$Matches[1]
                    }
                } catch {
                    Write-Host " ERROR: $_" -ForegroundColor Red
                }
            }
            
            # Run optimized trials
            $currentConfig++
            $optimizedTimes = @()
            $optimizedRanks = @()
            for ($t = 1; $t -le $Trials; $t++) {
                Write-Host "  Optimized trial $t/$Trials..." -NoNewline
                $trialSeed = $Seed + $t
                try {
                    $output = & .\optimized\quantum_sim.exe -n $n -d $Depth --initial-rank $rank --mode $mode --seed $trialSeed --allow-swap --non-interactive 2>&1
                    $outputStr = $output | Out-String
                    
                    $timePattern = 'Time:\s+([0-9.]+)\s*s'
                    if ($outputStr -match $timePattern) {
                        $time = [double]$Matches[1]
                        $optimizedTimes += $time
                        Write-Host " $([Math]::Round($time, 3))s" -ForegroundColor Green
                    } else {
                        Write-Host " (parse error)" -ForegroundColor Yellow
                    }
                    
                    $rankPattern = 'Final Rank:\s+([0-9]+)'
                    if ($outputStr -match $rankPattern) {
                        $optimizedRanks += [int]$Matches[1]
                    }
                } catch {
                    Write-Host " ERROR: $_" -ForegroundColor Red
                }
            }
            
            # Calculate statistics
            if ($baselineTimes.Count -gt 0 -and $optimizedTimes.Count -gt 0) {
                $baselineMean = ($baselineTimes | Measure-Object -Average).Average
                $optimizedMean = ($optimizedTimes | Measure-Object -Average).Average
                $speedup = $baselineMean / $optimizedMean
                
                $baselineRankMean = if ($baselineRanks.Count -gt 0) { ($baselineRanks | Measure-Object -Average).Average } else { 0 }
                $optimizedRankMean = if ($optimizedRanks.Count -gt 0) { ($optimizedRanks | Measure-Object -Average).Average } else { 0 }
                
                $result = [PSCustomObject]@{
                    Qubits = $n
                    Depth = $Depth
                    InitialRank = $rank
                    Mode = $mode
                    Seed = $Seed
                    Trials = $Trials
                    Baseline_Mean_s = [Math]::Round($baselineMean, 4)
                    Optimized_Mean_s = [Math]::Round($optimizedMean, 4)
                    Speedup = [Math]::Round($speedup, 3)
                    Baseline_FinalRank = [Math]::Round($baselineRankMean, 0)
                    Optimized_FinalRank = [Math]::Round($optimizedRankMean, 0)
                }
                $allResults += $result
                
                $color = if ($speedup -gt 1.05) { "Green" } elseif ($speedup -lt 0.95) { "Red" } else { "White" }
                Write-Host "  --> Speedup: $([Math]::Round($speedup, 2))x" -ForegroundColor $color
            }
            Write-Host ""
        }
    }
}

# Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║                     BENCHMARK RESULTS                          ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

$allResults | Format-Table -AutoSize

# Export to CSV
$csvPath = Join-Path $outPath "benchmark_results.csv"
$allResults | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "Results saved to: $csvPath"

# Generate summary statistics
Write-Host ""
Write-Host "═══ SUMMARY STATISTICS ═══"
$avgSpeedup = ($allResults | Measure-Object -Property Speedup -Average).Average
$maxSpeedup = ($allResults | Measure-Object -Property Speedup -Maximum).Maximum
$minSpeedup = ($allResults | Measure-Object -Property Speedup -Minimum).Minimum

Write-Host "Average Speedup: $([Math]::Round($avgSpeedup, 3))x"
Write-Host "Max Speedup: $([Math]::Round($maxSpeedup, 3))x"
Write-Host "Min Speedup: $([Math]::Round($minSpeedup, 3))x"

# Best/worst configurations
$best = $allResults | Sort-Object Speedup -Descending | Select-Object -First 1
$worst = $allResults | Sort-Object Speedup | Select-Object -First 1
Write-Host ""
Write-Host "Best config: n=$($best.Qubits), rank=$($best.InitialRank), mode=$($best.Mode) -> $($best.Speedup)x"
Write-Host "Worst config: n=$($worst.Qubits), rank=$($worst.InitialRank), mode=$($worst.Mode) -> $($worst.Speedup)x"

# Mode-specific analysis
Write-Host ""
Write-Host "═══ BY PARALLELIZATION MODE ═══"
$allResults | Group-Object Mode | ForEach-Object {
    $modeAvg = ($_.Group | Measure-Object -Property Speedup -Average).Average
    Write-Host "$($_.Name): $([Math]::Round($modeAvg, 3))x average speedup"
}

# Rank-specific analysis
Write-Host ""
Write-Host "═══ BY INITIAL RANK ═══"
$allResults | Group-Object InitialRank | Sort-Object { [int]$_.Name } | ForEach-Object {
    $rankAvg = ($_.Group | Measure-Object -Property Speedup -Average).Average
    Write-Host "Rank $($_.Name): $([Math]::Round($rankAvg, 3))x average speedup"
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════"
Write-Host "  BENCHMARK COMPLETE - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "═══════════════════════════════════════════════════════════════"
