#!/usr/bin/env python3
"""
CPU Usage Monitor - Tracks overall and per-core CPU usage
Monitors Python processes running benchmarks
Saves data to CSV file when log directory is provided
"""

import psutil
import time
import sys
import os
import csv
from datetime import datetime

def find_benchmark_process():
    """Find the Python process running a benchmark script."""
    benchmark_keywords = [
        'pennylane_', 'compare_all_', '4q_50e', '8q_100e', '8q_200e',
        'pennylane', 'qlret', '--worker'
    ]
    
    current_pid = psutil.Process().pid
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name'].lower():
                cmdline_str = ' '.join(str(arg) for arg in cmdline)
                
                # Check if this looks like a benchmark process
                if any(kw in cmdline_str.lower() for kw in benchmark_keywords):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            pass
    
    return None


def main():
    # Check for log directory argument
    log_dir = None
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print("CPU USAGE MONITOR")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    if log_dir:
        print(f"Log directory: {log_dir}")
    print("=" * 70)

    # Setup CSV file if log directory provided
    csv_file = None
    csv_writer = None
    if log_dir:
        csv_path = os.path.join(log_dir, "cpu_usage.csv")
        csv_file = open(csv_path, 'w', newline='')
        num_cores = psutil.cpu_count(logical=True)
        headers = ['timestamp', 'overall_cpu', 'process_cpu', 'process_status']
        headers += [f'core_{i}' for i in range(num_cores)]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        print(f"CSV logging to: {csv_path}")

    # Wait for benchmark process to start
    print("\nSearching for benchmark process...")
    benchmark_process = None
    search_attempts = 0
    max_attempts = 30  # Wait up to 60 seconds
    
    while benchmark_process is None and search_attempts < max_attempts:
        benchmark_process = find_benchmark_process()
        if benchmark_process is None:
            search_attempts += 1
            if search_attempts % 5 == 0:
                print(f"  Still searching... ({search_attempts * 2}s)")
            time.sleep(2)
    
    if benchmark_process:
        print(f"\n✓ Found benchmark process: PID={benchmark_process.pid}")
        try:
            cmdline = ' '.join(benchmark_process.cmdline()[:3])
            print(f"  Command: {cmdline}...")
        except:
            pass
    else:
        print("\n⚠ No benchmark process found after 60s. Monitoring system CPU only...")

    print("\nMonitoring CPU usage (updating every 2 seconds)...")
    print("Press Ctrl+C to stop monitoring\n")
    
    num_cores = psutil.cpu_count(logical=True)
    header = f"{'Time':<10} {'Overall':<9} {'Process':<10}"
    for i in range(num_cores):
        header += f" Core{i:<2}"
    print(header)
    print("-" * (30 + num_cores * 7))

    try:
        while True:
            timestamp = datetime.now().strftime('%H:%M:%S')
            timestamp_full = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Overall CPU usage
            overall_cpu = psutil.cpu_percent(interval=1)
            
            # Per-core CPU usage
            per_core = psutil.cpu_percent(interval=0, percpu=True)
            
            # Benchmark process CPU
            process_cpu = 0.0
            process_cpu_str = "N/A"
            process_status = "searching"
            
            if benchmark_process:
                try:
                    process_cpu = benchmark_process.cpu_percent(interval=0.5)
                    process_cpu_str = f"{process_cpu:>5.1f}%"
                    process_status = "running"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"\n✓ Benchmark process completed at {timestamp}")
                    process_status = "completed"
                    
                    # Save final entry
                    if csv_writer:
                        row = [timestamp_full, overall_cpu, 0.0, process_status] + list(per_core)
                        csv_writer.writerow(row)
                        csv_file.flush()
                    
                    print("\nFinal CPU stats recorded. Monitor will continue for 10 more seconds...")
                    for _ in range(5):
                        time.sleep(2)
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        timestamp_full = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        overall_cpu = psutil.cpu_percent(interval=1)
                        per_core = psutil.cpu_percent(interval=0, percpu=True)
                        cores_str = " ".join([f"{c:5.1f}%" for c in per_core])
                        print(f"{timestamp:<10} {overall_cpu:>5.1f}%   {'done':<10} {cores_str}")
                        
                        if csv_writer:
                            row = [timestamp_full, overall_cpu, 0.0, "done"] + list(per_core)
                            csv_writer.writerow(row)
                            csv_file.flush()
                    break
            
            # Save to CSV
            if csv_writer:
                row = [timestamp_full, overall_cpu, process_cpu, process_status] + list(per_core)
                csv_writer.writerow(row)
                csv_file.flush()
            
            cores_str = " ".join([f"{c:5.1f}%" for c in per_core])
            print(f"{timestamp:<10} {overall_cpu:>5.1f}%   {process_cpu_str:<10} {cores_str}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped by user")
    
    # Close CSV file
    if csv_file:
        csv_file.close()
        print(f"\n✓ CPU data saved to: {os.path.join(log_dir, 'cpu_usage.csv')}")
    
    print("=" * 70)
    print("Monitor session ended.")

if __name__ == "__main__":
    main()
