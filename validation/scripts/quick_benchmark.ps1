# LRET Quick Benchmark - Baseline vs Optimized
# Phase A.2: Simple benchmark comparison

param(
    [int]$Trials = 2,
    [int]$Seed = 42
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outPath = "results\$timestamp"
New-Item -ItemType Directory -Path $outPath -Force | Out-Null

Write-Host "LRET Quick Benchmark - $timestamp"
Write-Host "Trials: $Trials, Seed: $Seed"
Write-Host "Output: $outPath"
Write-Host ""

$configs = @(
    @{n=8; rank=8; mode="row"},
    @{n=8; rank=16; mode="row"},
    @{n=8; rank=32; mode="row"},
    @{n=10; rank=8; mode="row"},
    @{n=10; rank=16; mode="row"},
    @{n=10; rank=32; mode="row"},
    @{n=8; rank=16; mode="column"},
    @{n=10; rank=16; mode="column"}
)

$results = @()

foreach ($cfg in $configs) {
    $n = $cfg.n
    $rank = $cfg.rank
    $mode = $cfg.mode
    
    Write-Host "Config: n=$n, rank=$rank, mode=$mode"
    
    $baseTimes = @()
    $optTimes = @()
    
    for ($t = 1; $t -le $Trials; $t++) {
        $tseed = $Seed + $t
        
        # Baseline
        Write-Host "  Baseline $t..." -NoNewline
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & .\baseline\quantum_sim.exe -n $n -d 15 --initial-rank $rank --mode $mode --seed $tseed --allow-swap --non-interactive 2>&1
        $sw.Stop()
        $baseTimes += $sw.Elapsed.TotalSeconds
        Write-Host " $([Math]::Round($sw.Elapsed.TotalSeconds, 2))s"
        
        # Optimized
        Write-Host "  Optimized $t..." -NoNewline
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & .\optimized\quantum_sim.exe -n $n -d 15 --initial-rank $rank --mode $mode --seed $tseed --allow-swap --non-interactive 2>&1
        $sw.Stop()
        $optTimes += $sw.Elapsed.TotalSeconds
        Write-Host " $([Math]::Round($sw.Elapsed.TotalSeconds, 2))s"
    }
    
    $baseAvg = ($baseTimes | Measure-Object -Average).Average
    $optAvg = ($optTimes | Measure-Object -Average).Average
    $speedup = [Math]::Round($baseAvg / $optAvg, 2)
    
    $color = if ($speedup -gt 1.05) { "Green" } elseif ($speedup -lt 0.95) { "Red" } else { "White" }
    Write-Host "  -> Speedup: ${speedup}x" -ForegroundColor $color
    Write-Host ""
    
    $results += [PSCustomObject]@{
        Qubits = $n
        Rank = $rank
        Mode = $mode
        Baseline = [Math]::Round($baseAvg, 3)
        Optimized = [Math]::Round($optAvg, 3)
        Speedup = $speedup
    }
}

Write-Host "========== RESULTS =========="
$results | Format-Table -AutoSize

$csvPath = "$outPath\results.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "Saved to: $csvPath"

# Summary
$avgSpeedup = [Math]::Round(($results | Measure-Object -Property Speedup -Average).Average, 2)
Write-Host ""
Write-Host "Average Speedup: ${avgSpeedup}x"
