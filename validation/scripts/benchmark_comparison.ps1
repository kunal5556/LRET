# LRET Validation Benchmark: Baseline vs Optimized
# Phase A.1 Complete - This script compares baseline and optimized versions
# Run from D:\LRET\validation directory

param(
    [int]$Trials = 3,
    [int[]]$Qubits = @(8, 10, 12),
    [int]$Depth = 20,
    [int[]]$InitialRanks = @(4, 16, 32),
    [string]$OutputDir = "results"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Ensure output directories exist
New-Item -ItemType Directory -Path "$OutputDir\baseline" -Force | Out-Null
New-Item -ItemType Directory -Path "$OutputDir\optimized" -Force | Out-Null

Write-Host "============================================================"
Write-Host "      LRET Validation: Baseline vs Optimized Comparison     "
Write-Host "============================================================"
Write-Host "Timestamp: $timestamp"
Write-Host "Trials per configuration: $Trials"
Write-Host "Qubits: $($Qubits -join ', ')"
Write-Host "Depth: $Depth"
Write-Host "Initial Ranks: $($InitialRanks -join ', ')"
Write-Host "============================================================"
Write-Host ""

# Results storage
$results = @()

foreach ($n in $Qubits) {
    foreach ($rank in $InitialRanks) {
        Write-Host "`n>>> Configuration: $n qubits, rank=$rank, depth=$Depth"
        
        # Baseline runs
        $baselineTimes = @()
        for ($t = 1; $t -le $Trials; $t++) {
            Write-Host "  Baseline trial $t/$Trials..." -NoNewline
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $output = & .\baseline\quantum_sim.exe -n $n -d $Depth --initial-rank $rank --mode row --allow-swap --non-interactive 2>&1
            $sw.Stop()
            $baselineTimes += $sw.Elapsed.TotalSeconds
            Write-Host " $([Math]::Round($sw.Elapsed.TotalSeconds, 3))s"
        }
        
        # Optimized runs
        $optimizedTimes = @()
        for ($t = 1; $t -le $Trials; $t++) {
            Write-Host "  Optimized trial $t/$Trials..." -NoNewline
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $output = & .\optimized\quantum_sim.exe -n $n -d $Depth --initial-rank $rank --mode row --allow-swap --non-interactive 2>&1
            $sw.Stop()
            $optimizedTimes += $sw.Elapsed.TotalSeconds
            Write-Host " $([Math]::Round($sw.Elapsed.TotalSeconds, 3))s"
        }
        
        # Calculate statistics
        $baselineMean = ($baselineTimes | Measure-Object -Average).Average
        $optimizedMean = ($optimizedTimes | Measure-Object -Average).Average
        $speedup = [Math]::Round($baselineMean / $optimizedMean, 2)
        
        $result = [PSCustomObject]@{
            Qubits = $n
            InitialRank = $rank
            Depth = $Depth
            BaselineMean_s = [Math]::Round($baselineMean, 4)
            OptimizedMean_s = [Math]::Round($optimizedMean, 4)
            Speedup = $speedup
            Trials = $Trials
        }
        $results += $result
        
        Write-Host "  --> Baseline avg: $([Math]::Round($baselineMean, 4))s | Optimized avg: $([Math]::Round($optimizedMean, 4))s | Speedup: ${speedup}x"
    }
}

Write-Host "`n============================================================"
Write-Host "                      SUMMARY RESULTS                       "
Write-Host "============================================================"

$results | Format-Table -AutoSize

# Export to CSV
$csvPath = "$OutputDir\comparison_$timestamp.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "`nResults saved to: $csvPath"

# Generate summary
$summaryPath = "$OutputDir\summary_$timestamp.md"
@"
# LRET Validation Comparison Results

**Generated**: $timestamp
**Trials**: $Trials per configuration
**Depth**: $Depth gates

## Results

| Qubits | Initial Rank | Baseline (s) | Optimized (s) | Speedup |
|--------|--------------|--------------|---------------|---------|
"@ | Out-File $summaryPath

foreach ($r in $results) {
    "| $($r.Qubits) | $($r.InitialRank) | $($r.BaselineMean_s) | $($r.OptimizedMean_s) | $($r.Speedup)x |" | Out-File $summaryPath -Append
}

@"

## Key Findings

- Average speedup across all configurations: $([Math]::Round(($results | Measure-Object -Property Speedup -Average).Average, 2))x
- Best speedup: $($results | Sort-Object Speedup -Descending | Select-Object -First 1 | ForEach-Object { "$($_.Speedup)x at $($_.Qubits)q rank=$($_.InitialRank)" })

## Notes

This comparison uses ROW parallelization mode to directly test the Phase 1 optimization
(MIN_RANK_FOR_COL_PARALLEL threshold change from 4 to 32).
"@ | Out-File $summaryPath -Append

Write-Host "Summary saved to: $summaryPath"
Write-Host "`n=== BENCHMARK COMPLETE ==="
