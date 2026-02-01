#!/usr/bin/env python3
"""
CPU Usage Monitor - Tracks overall and per-core CPU usage
Shows ALL cores in a grouped display format
Monitors Python and LRET processes running benchmarks
Saves data to CSV file when log directory is provided
"""

import psutil
import time
import sys
import os
import csv
from datetime import datetime

def find_benchmark_processes():
    """Find Python and quantum_sim processes running benchmarks."""
    benchmark_keywords = [
        'parallel_modes', 'cirq_comparison', 'benchmark', 'run_benchmark',
        '--worker', 'quantum_sim'
    ]
    
    current_pid = psutil.Process().pid
    processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            
            cmdline = proc.info['cmdline']
            name = proc.info['name'].lower() if proc.info['name'] else ''
            
            # Check for quantum_sim directly
            if 'quantum_sim' in name:
                processes.append(proc)
                continue
            
            if cmdline and 'python' in name:
                cmdline_str = ' '.join(str(arg) for arg in cmdline)
                
                # Check if this looks like a benchmark process
                if any(kw in cmdline_str.lower() for kw in benchmark_keywords):
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            pass
    
    return processes


def format_core_grid(per_core, cores_per_row=16):
    """Format per-core usage as a grid display."""
    lines = []
    num_cores = len(per_core)
    
    for row_start in range(0, num_cores, cores_per_row):
        row_end = min(row_start + cores_per_row, num_cores)
        
        # Values row
        values = " ".join(f"{per_core[i]:>3.0f}" for i in range(row_start, row_end))
        
        lines.append(f"  Cores {row_start:>2}-{row_end-1:<2}: [{values}]")
    
    return lines


def get_color_indicator(value):
    """Get a text-based indicator for CPU usage level."""
    if value >= 90:
        return "█████"  # Very high
    elif value >= 70:
        return "████░"  # High
    elif value >= 50:
        return "███░░"  # Medium
    elif value >= 25:
        return "██░░░"  # Low
    elif value >= 10:
        return "█░░░░"  # Very low
    else:
        return "░░░░░"  # Idle


def main():
    # Check for log directory argument
    log_dir = None
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    num_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    print("=" * 80)
    print("          CPU USAGE MONITOR - LRET Parallel Modes Comparison")
    print("=" * 80)
    print(f"  Start time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Physical cores: {physical_cores}")
    print(f"  Logical cores : {num_cores}")
    print(f"  Total RAM     : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if log_dir:
        print(f"  Log directory : {log_dir}")
    print("=" * 80)

    # Setup CSV file if log directory provided
    csv_file = None
    csv_writer = None
    if log_dir:
        csv_path = os.path.join(log_dir, "cpu_usage.csv")
        csv_file = open(csv_path, 'w', newline='')
        headers = ['timestamp', 'elapsed_s', 'overall_cpu', 'active_cores', 
                   'total_process_cpu', 'total_memory_mb', 'num_processes']
        headers += [f'core_{i}' for i in range(num_cores)]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        print(f"  CSV logging to: {csv_path}")

    # Wait for benchmark process to start
    print("\n" + "-" * 80)
    print("  Searching for benchmark processes...")
    benchmark_processes = []
    search_attempts = 0
    max_attempts = 30  # Wait up to 60 seconds
    
    while not benchmark_processes and search_attempts < max_attempts:
        benchmark_processes = find_benchmark_processes()
        if not benchmark_processes:
            search_attempts += 1
            if search_attempts % 5 == 0:
                print(f"    Still searching... ({search_attempts * 2}s)")
            time.sleep(2)
    
    if benchmark_processes:
        print(f"\n  ✓ Found {len(benchmark_processes)} benchmark process(es):")
        for proc in benchmark_processes:
            try:
                name = proc.name()
                pid = proc.pid
                print(f"    - {name} (PID: {pid})")
            except:
                pass
    else:
        print("\n  ⚠ No benchmark process found after 60s. Monitoring system CPU only...")

    print("\n" + "=" * 80)
    print("  LIVE CPU MONITORING (updates every 2 seconds)")
    print("  Press Ctrl+C to stop")
    print("=" * 80)
    
    start_time = time.time()
    update_count = 0
    
    try:
        while True:
            update_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
            
            # Overall CPU usage
            overall_cpu = psutil.cpu_percent(interval=0.1)
            
            # Per-core CPU usage
            per_core = psutil.cpu_percent(percpu=True)
            
            # Count active cores (>5% usage)
            active_cores = sum(1 for c in per_core if c > 5)
            
            # Memory info
            mem = psutil.virtual_memory()
            
            # Process-specific stats
            total_process_cpu = 0.0
            total_memory_mb = 0.0
            process_details = []
            
            # Refresh process list periodically
            if update_count % 10 == 0:
                benchmark_processes = find_benchmark_processes()
            
            for proc in benchmark_processes[:]:  # Copy list to allow modification
                try:
                    cpu = proc.cpu_percent(interval=0.05)
                    mem_mb = proc.memory_info().rss / (1024**2)
                    status = proc.status()
                    name = proc.name()
                    total_process_cpu += cpu
                    total_memory_mb += mem_mb
                    process_details.append((name, proc.pid, cpu, mem_mb, status))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    benchmark_processes.remove(proc)
            
            # Clear screen effect - print separator
            print("\n" + "-" * 80)
            print(f"  [{current_time}] Elapsed: {elapsed_str} | Update #{update_count}")
            print("-" * 80)
            
            # Overall stats
            print(f"\n  SYSTEM OVERVIEW:")
            print(f"    Overall CPU   : {overall_cpu:>6.1f}%  {get_color_indicator(overall_cpu)}")
            print(f"    Active Cores  : {active_cores:>6d}/{num_cores} ({100*active_cores/num_cores:.0f}%)")
            print(f"    Memory Used   : {mem.used/(1024**3):>6.1f} GB / {mem.total/(1024**3):.1f} GB ({mem.percent:.0f}%)")
            
            # Process stats
            if process_details:
                print(f"\n  BENCHMARK PROCESSES ({len(process_details)} active):")
                for name, pid, cpu, mem_mb, status in process_details:
                    print(f"    {name:<15} PID:{pid:<8} CPU:{cpu:>6.1f}%  Mem:{mem_mb:>8.1f}MB  [{status}]")
                print(f"    {'─' * 60}")
                print(f"    {'TOTAL':<15} {'':8} CPU:{total_process_cpu:>6.1f}%  Mem:{total_memory_mb:>8.1f}MB")
            else:
                print(f"\n  BENCHMARK PROCESSES: None detected")
            
            # Per-core display - ALL CORES
            print(f"\n  PER-CORE CPU USAGE (%):")
            core_lines = format_core_grid(per_core, cores_per_row=16)
            for line in core_lines:
                print(line)
            
            # Core usage histogram
            print(f"\n  CORE UTILIZATION HISTOGRAM:")
            bins = [(0, 5, "Idle"), (5, 25, "Low"), (25, 50, "Med"), (50, 75, "High"), (75, 100, "Full")]
            for low, high, label in bins:
                count = sum(1 for c in per_core if low <= c < high)
                bar = "█" * (count * 40 // num_cores) if num_cores > 0 else ""
                print(f"    {label:>5} ({low:>2}-{high:<3}%): {count:>3} cores  |{bar:<40}|")
            
            # CSV logging
            if csv_writer:
                csv_row = [
                    datetime.now().isoformat(),
                    f"{elapsed:.1f}",
                    f"{overall_cpu:.1f}",
                    active_cores,
                    f"{total_process_cpu:.1f}",
                    f"{total_memory_mb:.1f}",
                    len(process_details)
                ]
                csv_row += [f"{c:.1f}" for c in per_core]
                csv_writer.writerow(csv_row)
                csv_file.flush()
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  ✓ Monitoring stopped by user")
    finally:
        if csv_file:
            csv_file.close()
            print(f"  ✓ CPU data saved to: {csv_path}")

    elapsed_total = time.time() - start_time
    hours = int(elapsed_total // 3600)
    mins = int((elapsed_total % 3600) // 60)
    secs = int(elapsed_total % 60)
    print(f"\n  Total monitoring time: {hours:02d}:{mins:02d}:{secs:02d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
