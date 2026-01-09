#!/bin/bash
# Monitor QNN Benchmark Progress

echo "==================================================================="
echo "QNN Benchmark Progress Monitor"
echo "==================================================================="
echo ""

# Check if process is running
if ps aux | grep "qnn_benchmark.py" | grep -v grep > /dev/null; then
    echo "✓ Benchmark is RUNNING"
    echo ""
    
    # Show process info
    echo "Process Info:"
    ps aux | grep "qnn_benchmark.py" | grep -v grep | awk '{printf "  PID: %s\n  CPU: %s%%\n  Memory: %s MB\n  Time: %s\n", $2, $3, $6/1024, $10}'
    echo ""
    
    # Show log statistics
    echo "Log Statistics:"
    cd /Users/suryanshsingh/Documents/LRET/results/qnn_benchmark
    echo "  Total lines: $(wc -l < benchmark.log)"
    echo "  File size: $(ls -lh benchmark.log | awk '{print $5}')"
    echo ""
    
    # Show latest output (last 30 lines)
    echo "Latest Output:"
    echo "-------------------------------------------------------------------"
    tail -30 benchmark.log
    echo "-------------------------------------------------------------------"
    echo ""
    
    # Check for results files
    echo "Output Files:"
    if [ -f "qnn_benchmark_results.json" ]; then
        echo "  ✓ qnn_benchmark_results.json ($(ls -lh qnn_benchmark_results.json | awk '{print $5}'))"
    else
        echo "  ⏳ qnn_benchmark_results.json (not yet created)"
    fi
    
    if [ -f "QNN_BENCHMARK_REPORT.md" ]; then
        echo "  ✓ QNN_BENCHMARK_REPORT.md ($(ls -lh QNN_BENCHMARK_REPORT.md | awk '{print $5}'))"
    else
        echo "  ⏳ QNN_BENCHMARK_REPORT.md (not yet created)"
    fi
    echo ""
    
    # Estimate progress (rough)
    epoch_count=$(grep -c "Epoch" benchmark.log || echo "0")
    total_epochs=40  # 10 epochs × 2 devices × 2 qubit counts
    if [ "$epoch_count" -gt 0 ]; then
        progress=$((epoch_count * 100 / total_epochs))
        echo "Estimated Progress: $epoch_count/$total_epochs epochs (~$progress%)"
    else
        echo "Estimated Progress: Initializing..."
    fi
    
else
    echo "✗ Benchmark is NOT running"
    echo ""
    
    # Check if completed
    cd /Users/suryanshsingh/Documents/LRET/results/qnn_benchmark
    if [ -f "QNN_BENCHMARK_REPORT.md" ]; then
        echo "✓✓ Benchmark COMPLETED!"
        echo ""
        echo "Results available in:"
        echo "  - qnn_benchmark_results.json"
        echo "  - QNN_BENCHMARK_REPORT.md"
        echo ""
        echo "To view report:"
        echo "  cat QNN_BENCHMARK_REPORT.md"
    else
        echo "⚠️ Benchmark stopped before completion"
        echo ""
        echo "Check log for errors:"
        echo "  tail -100 benchmark.log"
    fi
fi

echo ""
echo "==================================================================="
echo "To monitor continuously, run: watch -n 30 ./monitor_benchmark.sh"
echo "==================================================================="
