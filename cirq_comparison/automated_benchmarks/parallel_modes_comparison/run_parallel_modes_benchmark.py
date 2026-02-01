#!/usr/bin/env python3
"""
LRET Parallel Modes Comparison Benchmark
==========================================
Compares ALL LRET parallelization modes + Cirq FDM:

LRET Modes:
- SEQUENTIAL: Single-threaded baseline
- ROW: Row-wise parallel operations
- COLUMN: Column-wise parallel operations
- BATCH: Gate batching parallelism
- HYBRID: Combined row + batch (default)

Baseline:
- Cirq FDM (DensityMatrixSimulator)

This benchmark:
1. Opens TWO terminal windows (benchmark output + CPU monitor)
2. Tests each mode for various qubit counts
3. Generates comprehensive comparison plots
4. Creates detailed report

Configuration:
- Qubits: 8-30 (extended range, may OOM at high qubits)
- Depth: 25 CNOT layers
- Noise: 10% depolarizing per gate
- Epsilon: 1e-6
"""

import sys
import os
import time
import platform
from datetime import datetime

# Check if we're the launcher or the worker
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    IS_WORKER = True
else:
    IS_WORKER = False

# =============================================================================
# LAUNCHER MODE - Start benchmark and CPU monitor in separate windows
# =============================================================================
if not IS_WORKER:
    from launcher_utils import launch_in_new_terminal, get_terminal_name, format_command_for_platform
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Create unique results directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, 'results', f'{script_name}_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    terminal_name = get_terminal_name()
    
    print("=" * 70)
    print("LRET PARALLEL MODES COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Platform: {platform.system()}")
    print(f"Script: {os.path.basename(script_path)}")
    print(f"Results directory: {log_dir}")
    print("")
    print(f"This will open TWO new {terminal_name} windows:")
    print("  1. Benchmark execution window (main output)")
    print("  2. CPU monitoring window (resource tracking)")
    print("=" * 70)
    
    # Start benchmark in new window with log_dir argument
    benchmark_cmd = format_command_for_platform(script_path, "--worker", log_dir)
    launch_in_new_terminal(benchmark_cmd, "LRET Parallel Modes Benchmark")
    
    # Wait a moment for benchmark to start
    time.sleep(2)
    
    # Start CPU monitor in new window with log_dir argument
    monitor_path = os.path.join(script_dir, "monitor_cpu.py")
    monitor_cmd = format_command_for_platform(monitor_path, log_dir)
    launch_in_new_terminal(monitor_cmd, "CPU Monitor")
    
    print(f"\n✓ Both windows launched. Check the new {terminal_name} windows.")
    print(f"✓ Results will be saved to: {log_dir}")
    print("\nThis window can be closed. The benchmark runs in the new windows.")
    sys.exit(0)

# =============================================================================
# WORKER MODE - Actual benchmark execution
# =============================================================================

import json
import subprocess
import numpy as np
from pathlib import Path
import csv

# Check dependencies
try:
    import cirq
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import psutil
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Please run: python -m pip install cirq matplotlib numpy psutil")
    sys.exit(1)

# Get log directory from command line argument
LOG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'qubits': [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],  # Extended test range
    'depth': 25,                             # CNOT layers
    'noise_prob': 0.01,                      # 1% depolarizing noise per gate
    'epsilon': 1e-6,                         # Rank truncation
    'n_trials': 3,                           # Trials per config
}

# All modes to test (LRET parallel modes + Cirq baseline)
LRET_MODES = ['sequential', 'row', 'column', 'batch', 'hybrid']
ALL_MODES = LRET_MODES + ['cirq']

# Auto-detect paths
LRET_ROOT = Path(__file__).parent.parent.parent.parent
QUANTUM_SIM = LRET_ROOT / "build" / "Release" / "quantum_sim.exe"

# Setup logging
log_file = LOG_DIR / "benchmark.log"

def log(msg, level="INFO"):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {msg}"
    print(log_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_circuit_json(n_qubits, depth, noise_prob, epsilon, parallel_mode):
    """Build circuit JSON for LRET with specified parallel mode."""
    ops = []
    
    # Initial H layer with noise
    for i in range(n_qubits):
        ops.append({"name": "H", "wires": [i]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
    
    # CNOT layers with noise
    for d in range(depth):
        # Even layer
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
            if noise_prob > 0:
                ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
                ops.append({"name": "DEPOLARIZE", "wires": [i+1], "params": [noise_prob]})
        
        # Odd layer (alternating)
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
                if noise_prob > 0:
                    ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
                    ops.append({"name": "DEPOLARIZE", "wires": [i+1], "params": [noise_prob]})
    
    return {
        "circuit": {
            "num_qubits": n_qubits,
            "operations": ops
        },
        "config": {
            "epsilon": epsilon,
            "initial_rank": 1,
            "parallel_mode": parallel_mode.upper()
        }
    }

def build_circuit_cirq(n_qubits, depth, noise_prob):
    """Build equivalent Cirq circuit."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    
    # Initial H layer with noise
    for i in range(n_qubits):
        ops.append(cirq.H(qubits[i]))
        if noise_prob > 0:
            ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
    
    # CNOT layers with noise
    for d in range(depth):
        for i in range(0, n_qubits - 1, 2):
            ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
            if noise_prob > 0:
                ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
        
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
                if noise_prob > 0:
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
    
    return cirq.Circuit(ops)

# =============================================================================
# BENCHMARK RUNNERS
# =============================================================================

def run_lret(circuit_json):
    """Run LRET simulation using quantum_sim.exe."""
    circuit_file = LOG_DIR / "temp_circuit.json"
    output_file = LOG_DIR / "temp_output.json"
    
    with open(circuit_file, 'w') as f:
        json.dump(circuit_json, f)
    
    try:
        result = subprocess.run(
            [str(QUANTUM_SIM), "--input-json", str(circuit_file), 
             "--output-json", str(output_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None, f"Non-zero exit code: {result.returncode}"
        
        if not output_file.exists():
            return None, "No output file generated"
        
        with open(output_file, 'r') as f:
            output = json.load(f)
        
        return output, None
    
    except Exception as e:
        return None, str(e)

def run_cirq(circuit):
    """Run Cirq simulation and compute metrics."""
    sim = cirq.DensityMatrixSimulator()
    try:
        result = sim.simulate(circuit)
        dm = result.final_density_matrix
        
        # Calculate metrics
        trace = np.trace(dm).real
        purity = np.trace(dm @ dm).real
        
        return {
            'density_matrix': dm,
            'trace': trace,
            'purity': purity,
        }, None
    except MemoryError as e:
        return None, f"Out of memory: {e}"
    except Exception as e:
        return None, str(e)

def benchmark_mode(mode, n_qubits, depth, noise_prob, epsilon, n_trials, test_num, total_tests, benchmark_start_time):
    """Run benchmark for single mode and qubit count with continuous output."""
    times = []
    rank = None
    status = None
    memory_mb = None
    trace_val = None
    purity = None
    execution_time_ms = None
    
    if mode == 'cirq':
        log(f"    Building Cirq circuit ({n_qubits}q, depth={depth})...")
        circuit = build_circuit_cirq(n_qubits, depth, noise_prob)
        
        for trial in range(n_trials):
            trial_start = time.perf_counter()
            
            # Continuous progress output
            elapsed_total = time.time() - benchmark_start_time
            completed = test_num - 1 + (trial / n_trials)
            if completed > 0:
                eta = (elapsed_total / completed) * (total_tests - completed)
                eta_str = f", ETA: {eta/60:.1f}min" if eta > 60 else f", ETA: {eta:.0f}s"
            else:
                eta_str = ""
            
            log(f"    Trial {trial+1}/{n_trials} running...{eta_str}")
            sys.stdout.flush()
            
            result, error = run_cirq(circuit)
            elapsed = (time.perf_counter() - trial_start) * 1000
            
            if error:
                log(f"    ✗ Cirq FAILED: {error}", "ERROR")
                return {
                    'mode': mode,
                    'n_qubits': n_qubits,
                    'depth': depth,
                    'noise_prob': noise_prob,
                    'epsilon': epsilon,
                    'time_mean': None,
                    'time_std': None,
                    'times': times,
                    'rank': None,
                    'status': f'FAILED: {error}',
                    'memory_mb': None,
                    'trace': None,
                    'purity': None,
                    'execution_time_ms': None,
                    'error': error,
                }
            
            times.append(elapsed)
            log(f"    Trial {trial+1}/{n_trials} completed: {elapsed:.2f}ms")
            
            # Get metrics from first trial
            if trial == 0:
                trace_val = result.get('trace', None)
                purity = result.get('purity', None)
        
        status = "completed"
        rank = 2**n_qubits  # Full density matrix
        memory_mb = (2**n_qubits)**2 * 16 / 1e6  # Complex double
    
    else:
        log(f"    Building LRET circuit ({n_qubits}q, depth={depth}, mode={mode.upper()})...")
        circuit = build_circuit_json(n_qubits, depth, noise_prob, epsilon, mode)
        
        for trial in range(n_trials):
            trial_start = time.perf_counter()
            
            # Continuous progress output
            elapsed_total = time.time() - benchmark_start_time
            completed = test_num - 1 + (trial / n_trials)
            if completed > 0:
                eta = (elapsed_total / completed) * (total_tests - completed)
                eta_str = f", ETA: {eta/60:.1f}min" if eta > 60 else f", ETA: {eta:.0f}s"
            else:
                eta_str = ""
            
            log(f"    Trial {trial+1}/{n_trials} running...{eta_str}")
            sys.stdout.flush()
            
            result, error = run_lret(circuit)
            elapsed = (time.perf_counter() - trial_start) * 1000
            
            if error:
                log(f"    ✗ LRET ({mode}) FAILED: {error}", "ERROR")
                return {
                    'mode': mode,
                    'n_qubits': n_qubits,
                    'depth': depth,
                    'noise_prob': noise_prob,
                    'epsilon': epsilon,
                    'time_mean': None,
                    'time_std': None,
                    'times': times,
                    'rank': None,
                    'status': f'FAILED: {error}',
                    'memory_mb': None,
                    'trace': None,
                    'purity': None,
                    'execution_time_ms': None,
                    'error': error,
                }
            
            times.append(elapsed)
            log(f"    Trial {trial+1}/{n_trials} completed: {elapsed:.2f}ms")
            
            # Get metrics from first trial
            if trial == 0:
                status = result.get('status', 'unknown')
                rank = result.get('final_rank', None)
                execution_time_ms = result.get('execution_time_ms', None)
                # Estimate memory: rank * 2^n * 16 bytes (complex double)
                if rank:
                    memory_mb = rank * 2**n_qubits * 16 / 1e6
    
    return {
        'mode': mode,
        'n_qubits': n_qubits,
        'depth': depth,
        'noise_prob': noise_prob,
        'epsilon': epsilon,
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'times': times,
        'rank': rank,
        'status': status,
        'memory_mb': memory_mb,
        'trace': trace_val,
        'purity': purity,
        'execution_time_ms': execution_time_ms,
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_plots(results):
    """Generate comprehensive comparison plots."""
    log("\nGenerating plots...")
    
    # Filter to only successful results with valid time data
    valid_results = [r for r in results if r and r.get('time_mean') is not None]
    
    if not valid_results:
        log("No valid results to plot!", "WARNING")
        return
    
    # Organize data by mode
    mode_data = {mode: [] for mode in ALL_MODES}
    for r in valid_results:
        mode_data[r['mode']].append(r)
    
    # Sort by qubits
    for mode in ALL_MODES:
        mode_data[mode].sort(key=lambda x: x['n_qubits'])
    
    # Log what we're plotting
    for mode in ALL_MODES:
        if mode_data[mode]:
            qubits = [d['n_qubits'] for d in mode_data[mode]]
            log(f"  {mode.upper()}: data for {len(qubits)} qubit configs ({min(qubits)}-{max(qubits)})")
    
    # Colors for each mode
    colors = {
        'sequential': '#1f77b4',  # Blue
        'row': '#ff7f0e',         # Orange
        'column': '#2ca02c',      # Green
        'batch': '#d62728',       # Red
        'hybrid': '#9467bd',      # Purple
        'cirq': '#8c564b',        # Brown
    }
    
    markers = {
        'sequential': 'o',
        'row': 's',
        'column': '^',
        'batch': 'D',
        'hybrid': 'p',
        'cirq': 'X',
    }
    
    # Plot 1: Time comparison (all modes)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for mode in ALL_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            times = [d['time_mean'] for d in data]
            stds = [d['time_std'] for d in data]
            ax.errorbar(qubits, times, yerr=stds, fmt=f'{markers[mode]}-',
                       color=colors[mode], label=mode.upper(), 
                       capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'LRET Parallel Modes vs Cirq FDM\n(depth={CONFIG["depth"]}, noise={CONFIG["noise_prob"]*100:.2f}%)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'time_comparison_all.png', dpi=300)
    log("  ✓ time_comparison_all.png")
    plt.close()
    
    # Plot 2: Speedup over sequential (LRET modes only)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get sequential times as baseline
    seq_times = {d['n_qubits']: d['time_mean'] for d in mode_data['sequential']}
    
    for mode in LRET_MODES:
        if mode == 'sequential':
            continue
        data = mode_data[mode]
        if data:
            qubits = []
            speedups = []
            for d in data:
                if d['n_qubits'] in seq_times:
                    qubits.append(d['n_qubits'])
                    speedups.append(seq_times[d['n_qubits']] / d['time_mean'])
            
            if qubits:
                ax.plot(qubits, speedups, f'{markers[mode]}-', color=colors[mode],
                       label=mode.upper(), linewidth=2, markersize=8)
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Sequential (baseline)')
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title('LRET Parallel Modes Speedup over Sequential', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'speedup_vs_sequential.png', dpi=300)
    log("  ✓ speedup_vs_sequential.png")
    plt.close()
    
    # Plot 3: Speedup over Cirq (all LRET modes)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cirq_times = {d['n_qubits']: d['time_mean'] for d in mode_data['cirq']}
    
    for mode in LRET_MODES:
        data = mode_data[mode]
        if data:
            qubits = []
            speedups = []
            for d in data:
                if d['n_qubits'] in cirq_times:
                    qubits.append(d['n_qubits'])
                    speedups.append(cirq_times[d['n_qubits']] / d['time_mean'])
            
            if qubits:
                ax.plot(qubits, speedups, f'{markers[mode]}-', color=colors[mode],
                       label=mode.upper(), linewidth=2, markersize=8)
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Cirq (baseline)')
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup over Cirq (x)', fontsize=12, fontweight='bold')
    ax.set_title('LRET Speedup over Cirq FDM', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'speedup_vs_cirq.png', dpi=300)
    log("  ✓ speedup_vs_cirq.png")
    plt.close()
    
    # Plot 4: Memory comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for mode in ALL_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            memory = [d['memory_mb'] if d['memory_mb'] else 0 for d in data]
            if any(memory):
                ax.plot(qubits, memory, f'{markers[mode]}-', color=colors[mode],
                       label=mode.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'memory_comparison.png', dpi=300)
    log("  ✓ memory_comparison.png")
    plt.close()
    
    # Plot 5: Rank evolution (LRET only)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for mode in LRET_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            ranks = [d['rank'] if d['rank'] else 0 for d in data]
            if any(ranks):
                ax.plot(qubits, ranks, f'{markers[mode]}-', color=colors[mode],
                       label=mode.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Rank', fontsize=12, fontweight='bold')
    ax.set_title('LRET Rank Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'rank_evolution.png', dpi=300)
    log("  ✓ rank_evolution.png")
    plt.close()
    
    # Plot 6: Comprehensive 2x3 summary
    fig = plt.figure(figsize=(18, 12))
    
    # Time comparison
    ax1 = plt.subplot(2, 3, 1)
    for mode in ALL_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            times = [d['time_mean'] for d in data]
            ax1.plot(qubits, times, f'{markers[mode]}-', color=colors[mode],
                    label=mode.upper(), linewidth=2, markersize=6)
    ax1.set_yscale('log')
    ax1.set_xlabel('Qubits')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Speedup vs sequential
    ax2 = plt.subplot(2, 3, 2)
    for mode in LRET_MODES:
        if mode == 'sequential':
            continue
        data = mode_data[mode]
        if data:
            qubits = []
            speedups = []
            for d in data:
                if d['n_qubits'] in seq_times:
                    qubits.append(d['n_qubits'])
                    speedups.append(seq_times[d['n_qubits']] / d['time_mean'])
            if qubits:
                ax2.plot(qubits, speedups, f'{markers[mode]}-', color=colors[mode],
                        label=mode.upper(), linewidth=2, markersize=6)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Qubits')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs Sequential')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Speedup vs Cirq
    ax3 = plt.subplot(2, 3, 3)
    for mode in LRET_MODES:
        data = mode_data[mode]
        if data:
            qubits = []
            speedups = []
            for d in data:
                if d['n_qubits'] in cirq_times:
                    qubits.append(d['n_qubits'])
                    speedups.append(cirq_times[d['n_qubits']] / d['time_mean'])
            if qubits:
                ax3.plot(qubits, speedups, f'{markers[mode]}-', color=colors[mode],
                        label=mode.upper(), linewidth=2, markersize=6)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Qubits')
    ax3.set_ylabel('Speedup (x)')
    ax3.set_title('Speedup vs Cirq')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Memory
    ax4 = plt.subplot(2, 3, 4)
    for mode in ALL_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            memory = [d['memory_mb'] if d['memory_mb'] else 0 for d in data]
            if any(memory):
                ax4.plot(qubits, memory, f'{markers[mode]}-', color=colors[mode],
                        label=mode.upper(), linewidth=2, markersize=6)
    ax4.set_yscale('log')
    ax4.set_xlabel('Qubits')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('Memory Usage')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Rank
    ax5 = plt.subplot(2, 3, 5)
    for mode in LRET_MODES:
        data = mode_data[mode]
        if data:
            qubits = [d['n_qubits'] for d in data]
            ranks = [d['rank'] if d['rank'] else 0 for d in data]
            if any(ranks):
                ax5.plot(qubits, ranks, f'{markers[mode]}-', color=colors[mode],
                        label=mode.upper(), linewidth=2, markersize=6)
    ax5.set_xlabel('Qubits')
    ax5.set_ylabel('Final Rank')
    ax5.set_title('LRET Rank')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Bar chart: Best mode per qubit count
    ax6 = plt.subplot(2, 3, 6)
    qubits_tested = sorted(set(r['n_qubits'] for r in results if r))
    best_modes = []
    best_speedups = []
    for q in qubits_tested:
        best = None
        best_time = float('inf')
        for r in results:
            if r and r['n_qubits'] == q and r['mode'] in LRET_MODES:
                if r['time_mean'] < best_time:
                    best_time = r['time_mean']
                    best = r['mode']
        if best and q in cirq_times:
            best_modes.append(best)
            best_speedups.append(cirq_times[q] / best_time)
    
    if best_speedups:
        bar_colors = [colors[m] for m in best_modes]
        bars = ax6.bar(range(len(qubits_tested)), best_speedups, color=bar_colors)
        ax6.set_xticks(range(len(qubits_tested)))
        ax6.set_xticklabels([str(q) for q in qubits_tested])
        ax6.set_xlabel('Qubits')
        ax6.set_ylabel('Best LRET Speedup vs Cirq')
        ax6.set_title('Best Mode per Qubit Count')
        ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        # Add labels
        for i, (bar, mode) in enumerate(zip(bars, best_modes)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    mode[:3].upper(), ha='center', va='bottom', fontsize=7)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'LRET Parallel Modes Comprehensive Comparison\n' +
                 f'depth={CONFIG["depth"]}, noise={CONFIG["noise_prob"]*100:.2f}%, ε={CONFIG["epsilon"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(LOG_DIR / 'comprehensive_summary.png', dpi=300)
    log("  ✓ comprehensive_summary.png")
    plt.close()

def generate_report(results):
    """Generate detailed markdown report."""
    log("\nGenerating report...")
    
    # Organize data
    mode_data = {mode: [] for mode in ALL_MODES}
    for r in results:
        if r:
            mode_data[r['mode']].append(r)
    
    report = f"""# LRET Parallel Modes Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Qubits tested:** {CONFIG['qubits']}
- **Depth:** {CONFIG['depth']} CNOT layers
- **Noise:** {CONFIG['noise_prob']*100:.4f}% depolarizing per gate
- **Epsilon:** {CONFIG['epsilon']}
- **Trials per config:** {CONFIG['n_trials']}
- **Timeout:** {CONFIG['timeout']}s

## Modes Tested

| Mode | Description |
|------|-------------|
| SEQUENTIAL | Single-threaded baseline |
| ROW | Row-wise parallel operations |
| COLUMN | Column-wise parallel operations |
| BATCH | Gate batching parallelism |
| HYBRID | Combined row + batch (LRET default) |
| CIRQ | Cirq DensityMatrixSimulator (baseline) |

## Results Summary

### Execution Time (ms)

| Qubits | Sequential | Row | Column | Batch | Hybrid | Cirq |
|--------|------------|-----|--------|-------|--------|------|
"""
    
    # Build results table
    qubits_tested = sorted(set(r['n_qubits'] for r in results if r))
    for q in qubits_tested:
        row = f"| {q} |"
        for mode in ALL_MODES:
            data = [r for r in mode_data[mode] if r['n_qubits'] == q]
            if data:
                row += f" {data[0]['time_mean']:.1f}±{data[0]['time_std']:.1f} |"
            else:
                row += " FAIL |"
        report += row + "\n"
    
    # Calculate best modes
    report += "\n### Best LRET Mode per Qubit Count\n\n"
    report += "| Qubits | Best Mode | Time (ms) | Speedup vs Cirq |\n"
    report += "|--------|-----------|-----------|------------------|\n"
    
    cirq_times = {d['n_qubits']: d['time_mean'] for d in mode_data['cirq']}
    
    for q in qubits_tested:
        best = None
        best_time = float('inf')
        for mode in LRET_MODES:
            for r in mode_data[mode]:
                if r['n_qubits'] == q and r['time_mean'] < best_time:
                    best_time = r['time_mean']
                    best = mode
        
        if best and q in cirq_times:
            speedup = cirq_times[q] / best_time
            report += f"| {q} | {best.upper()} | {best_time:.1f} | {speedup:.2f}× |\n"
        elif best:
            report += f"| {q} | {best.upper()} | {best_time:.1f} | N/A |\n"
    
    # Average speedups
    report += "\n### Average Speedups\n\n"
    
    seq_times = {d['n_qubits']: d['time_mean'] for d in mode_data['sequential']}
    
    for mode in LRET_MODES + ['cirq']:
        if mode == 'sequential':
            continue
        
        speedups_vs_seq = []
        speedups_vs_cirq = []
        
        for r in mode_data[mode]:
            q = r['n_qubits']
            if q in seq_times:
                speedups_vs_seq.append(seq_times[q] / r['time_mean'])
            if q in cirq_times and mode != 'cirq':
                speedups_vs_cirq.append(cirq_times[q] / r['time_mean'])
        
        if speedups_vs_seq:
            avg_seq = np.mean(speedups_vs_seq)
            if mode == 'cirq':
                report += f"- **{mode.upper()}** vs Sequential: {avg_seq:.2f}×\n"
            else:
                if speedups_vs_cirq:
                    avg_cirq = np.mean(speedups_vs_cirq)
                    report += f"- **{mode.upper()}**: {avg_seq:.2f}× vs Sequential, {avg_cirq:.2f}× vs Cirq\n"
                else:
                    report += f"- **{mode.upper()}**: {avg_seq:.2f}× vs Sequential\n"
    
    report += """
## Plots Generated

1. `time_comparison_all.png` - Execution time all modes
2. `speedup_vs_sequential.png` - LRET mode speedups vs sequential
3. `speedup_vs_cirq.png` - LRET speedups vs Cirq
4. `memory_comparison.png` - Memory usage comparison
5. `rank_evolution.png` - LRET rank evolution
6. `comprehensive_summary.png` - 2×3 summary grid

## CPU Usage

See `cpu_usage.csv` for per-second CPU monitoring data.

## Raw Data

See `results.json` and `results.csv` for complete raw data.
"""
    
    with open(LOG_DIR / 'REPORT.md', 'w') as f:
        f.write(report)
    
    log("  ✓ REPORT.md")

def save_results_csv(results):
    """Save results to CSV file with all metrics."""
    csv_path = LOG_DIR / 'results.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'n_qubits', 'depth', 'noise_prob', 'epsilon', 
                        'time_mean_ms', 'time_std_ms', 'rank', 'status', 'memory_mb',
                        'trace', 'purity', 'execution_time_ms'])
        
        for r in results:
            if r:
                writer.writerow([
                    r['mode'], 
                    r['n_qubits'], 
                    r['depth'], 
                    r['noise_prob'],
                    r['epsilon'], 
                    f"{r['time_mean']:.2f}" if r.get('time_mean') else 'FAILED', 
                    f"{r['time_std']:.2f}" if r.get('time_std') else '',
                    r.get('rank', ''), 
                    r.get('status', 'unknown'), 
                    f"{r['memory_mb']:.2f}" if r.get('memory_mb') else '',
                    f"{r['trace']:.6f}" if r.get('trace') else '',
                    f"{r['purity']:.6f}" if r.get('purity') else '',
                    f"{r['execution_time_ms']:.2f}" if r.get('execution_time_ms') else '',
                ])
    
    log(f"  ✓ results.csv")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main benchmark execution."""
    benchmark_start_time = time.time()
    
    log("=" * 70)
    log("LRET PARALLEL MODES COMPARISON BENCHMARK")
    log("=" * 70)
    log(f"LRET Root: {LRET_ROOT}")
    log(f"quantum_sim.exe: {QUANTUM_SIM}")
    log(f"Output directory: {LOG_DIR}")
    log(f"")
    log(f"Configuration:")
    log(f"  Qubits: {CONFIG['qubits']} ({len(CONFIG['qubits'])} values)")
    log(f"  Depth: {CONFIG['depth']} CNOT layers")
    log(f"  Noise: {CONFIG['noise_prob']*100:.2f}% depolarizing per gate")
    log(f"  Epsilon: {CONFIG['epsilon']}")
    log(f"  Trials: {CONFIG['n_trials']} per configuration")
    log(f"  Modes: {', '.join(m.upper() for m in ALL_MODES)}")
    log(f"  CPU Cores: {os.cpu_count()}")
    log(f"")
    
    total_tests = len(CONFIG['qubits']) * len(ALL_MODES)
    log(f"Total test configurations: {total_tests}")
    log(f"Total individual runs: {total_tests * CONFIG['n_trials']}")
    log(f"")
    log(f"NOTE: High qubit counts (24+) may cause OOM failures - this is expected!")
    log(f"      The benchmark will continue and record which configurations fail.")
    
    # Verify quantum_sim.exe exists
    if not QUANTUM_SIM.exists():
        log(f"ERROR: quantum_sim.exe not found at {QUANTUM_SIM}", "ERROR")
        log("Please run setup script first", "ERROR")
        sys.exit(1)
    
    log(f"\n{'='*70}")
    log("STARTING BENCHMARK...")
    log(f"{'='*70}")
    
    # Run benchmarks
    results = []
    test_num = 0
    
    for n_qubits in CONFIG['qubits']:
        log(f"\n{'='*70}")
        log(f"Testing {n_qubits} qubits")
        log(f"{'='*70}")
        
        for mode in ALL_MODES:
            test_num += 1
            elapsed_total = time.time() - benchmark_start_time
            
            log(f"\n  [{test_num}/{total_tests}] Mode: {mode.upper()} | Elapsed: {elapsed_total/60:.1f}min")
            
            result = benchmark_mode(
                mode=mode,
                n_qubits=n_qubits,
                depth=CONFIG['depth'],
                noise_prob=CONFIG['noise_prob'],
                epsilon=CONFIG['epsilon'],
                n_trials=CONFIG['n_trials'],
                test_num=test_num,
                total_tests=total_tests,
                benchmark_start_time=benchmark_start_time
            )
            
            if result and result.get('time_mean') is not None:
                log(f"    ✓ COMPLETED: {result['time_mean']:.2f}±{result['time_std']:.2f}ms" +
                    (f", rank={result['rank']}" if result.get('rank') else "") +
                    (f", purity={result['purity']:.6f}" if result.get('purity') else ""))
            else:
                status = result.get('status', 'UNKNOWN') if result else 'FAILED'
                log(f"    ✗ {status}")
            
            results.append(result)
            
            # Save intermediate results (in case of crash)
            with open(LOG_DIR / 'results.json', 'w') as f:
                json.dump([r for r in results if r], f, indent=2, default=str)
    
    # Generate outputs
    generate_plots(results)
    generate_report(results)
    save_results_csv(results)
    
    log("\n" + "=" * 70)
    log("BENCHMARK COMPLETE!")
    log("=" * 70)
    log(f"Results saved to: {LOG_DIR}")
    log("")
    log("Files generated:")
    log("  - benchmark.log (this log)")
    log("  - results.json (raw data)")
    log("  - results.csv (tabular data)")
    log("  - REPORT.md (summary report)")
    log("  - *.png (visualization plots)")
    log("")

if __name__ == "__main__":
    main()
