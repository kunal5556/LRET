#!/usr/bin/env python3
"""
LRET Automated Benchmark Runner
================================

Configuration:
- Qubits: 10-20
- Depth: 20
- Noise: 0.01% depolarizing per gate
- Epsilon: 1e-6
- Comparison: LRET vs Cirq FDM (DensityMatrixSimulator)

This script runs the complete benchmark suite and generates all plots.
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Check dependencies
try:
    import cirq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import psutil
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Please run: python -m pip install cirq matplotlib numpy psutil")
    sys.exit(1)

# Configuration
CONFIG = {
    'qubits': [10, 12, 14, 16, 18, 20],  # Test range
    'depth': 20,
    'noise_prob': 0.0001,  # 0.01%
    'epsilon': 1e-6,
    'n_trials': 3,
    'timeout': 300,  # 5 minutes per simulation
}

LRET_ROOT = Path("d:/LRET")
QUANTUM_SIM = LRET_ROOT / "build" / "Release" / "quantum_sim.exe"
OUTPUT_DIR = Path(__file__).parent / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
log_file = OUTPUT_DIR / "benchmark.log"

def log(msg, level="INFO"):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {msg}"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

def build_circuit_json(n_qubits, depth, noise_prob, epsilon):
    """Build circuit JSON for LRET."""
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
            "initial_rank": 1
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
        # Even layer
        for i in range(0, n_qubits - 1, 2):
            ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
            if noise_prob > 0:
                ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
        
        # Odd layer
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
                if noise_prob > 0:
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
    
    return cirq.Circuit(ops)

def run_lret(circuit_json, timeout=300):
    """Run LRET simulation using quantum_sim.exe."""
    # Write circuit to temp file
    circuit_file = OUTPUT_DIR / "temp_circuit.json"
    output_file = OUTPUT_DIR / "temp_output.json"
    
    with open(circuit_file, 'w') as f:
        json.dump(circuit_json, f)
    
    # Run quantum_sim.exe
    try:
        result = subprocess.run(
            [str(QUANTUM_SIM), "--input-json", str(circuit_file), 
             "--output", str(output_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            return None, f"Non-zero exit code: {result.returncode}"
        
        if not output_file.exists():
            return None, "No output file generated"
        
        with open(output_file, 'r') as f:
            output = json.load(f)
        
        return output, None
    
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s"
    except Exception as e:
        return None, str(e)

def run_cirq(circuit, timeout=300):
    """Run Cirq simulation."""
    sim = cirq.DensityMatrixSimulator()
    try:
        result = sim.simulate(circuit)
        return result, None
    except Exception as e:
        return None, str(e)

def benchmark_single(n_qubits, depth, noise_prob, epsilon, n_trials, timeout):
    """Run benchmark for single configuration."""
    log(f"\n{'='*70}")
    log(f"Testing {n_qubits} qubits, depth={depth}, noise={noise_prob*100:.2f}%")
    log(f"{'='*70}")
    
    # Build circuits
    lret_circuit = build_circuit_json(n_qubits, depth, noise_prob, epsilon)
    cirq_circuit = build_circuit_cirq(n_qubits, depth, noise_prob)
    
    # LRET benchmarks
    log("Running LRET...")
    lret_times = []
    lret_status = None
    lret_rank = None
    
    for trial in range(n_trials):
        log(f"  Trial {trial+1}/{n_trials}...", "DEBUG")
        start = time.perf_counter()
        result, error = run_lret(lret_circuit, timeout)
        elapsed = (time.perf_counter() - start) * 1000
        
        if error:
            log(f"  LRET failed: {error}", "ERROR")
            break
        
        lret_times.append(elapsed)
        if trial == 0:
            lret_status = result.get('status', 'unknown')
            lret_rank = result.get('final_rank', None)
    
    if not lret_times:
        log("  LRET: FAILED", "ERROR")
        return None
    
    lret_mean = np.mean(lret_times)
    lret_std = np.std(lret_times)
    log(f"  LRET: {lret_mean:.2f}±{lret_std:.2f}ms, rank={lret_rank}, status={lret_status}")
    
    # Cirq benchmarks
    log("Running Cirq FDM...")
    cirq_times = []
    
    for trial in range(n_trials):
        log(f"  Trial {trial+1}/{n_trials}...", "DEBUG")
        start = time.perf_counter()
        result, error = run_cirq(cirq_circuit, timeout)
        elapsed = (time.perf_counter() - start) * 1000
        
        if error:
            log(f"  Cirq failed: {error}", "ERROR")
            break
        
        cirq_times.append(elapsed)
    
    if not cirq_times:
        log("  Cirq: FAILED", "ERROR")
        cirq_mean = None
        cirq_std = None
        speedup = None
    else:
        cirq_mean = np.mean(cirq_times)
        cirq_std = np.std(cirq_times)
        speedup = cirq_mean / lret_mean
        log(f"  Cirq: {cirq_mean:.2f}±{cirq_std:.2f}ms")
        log(f"  Speedup: {speedup:.2f}x")
    
    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'noise_prob': noise_prob,
        'epsilon': epsilon,
        'lret_mean': lret_mean,
        'lret_std': lret_std,
        'lret_rank': lret_rank,
        'lret_status': lret_status,
        'cirq_mean': cirq_mean,
        'cirq_std': cirq_std,
        'speedup': speedup,
    }

def generate_plots(results):
    """Generate all visualization plots."""
    log("\nGenerating plots...")
    
    # Filter successful results
    results = [r for r in results if r is not None and r['cirq_mean'] is not None]
    
    if not results:
        log("No successful results to plot", "ERROR")
        return
    
    qubits = [r['n_qubits'] for r in results]
    lret_times = [r['lret_mean'] for r in results]
    lret_stds = [r['lret_std'] for r in results]
    cirq_times = [r['cirq_mean'] for r in results]
    cirq_stds = [r['cirq_std'] for r in results]
    speedups = [r['speedup'] for r in results]
    ranks = [r['lret_rank'] for r in results]
    
    # Plot 1: Time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(qubits, lret_times, yerr=lret_stds, fmt='o-', label='LRET', 
                capsize=5, linewidth=2, markersize=8)
    ax.errorbar(qubits, cirq_times, yerr=cirq_stds, fmt='s-', label='Cirq FDM',
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Execution Time (depth={CONFIG["depth"]}, noise={CONFIG["noise_prob"]*100:.2f}%)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_comparison.png', dpi=300)
    log("  ✓ time_comparison.png")
    plt.close()
    
    # Plot 2: Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(qubits, speedups, 'o-', color='green', linewidth=3, markersize=10)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title('LRET Speedup over Cirq FDM', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for q, s in zip(qubits, speedups):
        ax.text(q, s, f'{s:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'speedup.png', dpi=300)
    log("  ✓ speedup.png")
    plt.close()
    
    # Plot 3: Rank evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(qubits, ranks, 'o-', color='purple', linewidth=3, markersize=10)
    ax.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Rank', fontsize=12, fontweight='bold')
    ax.set_title('LRET Rank Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rank.png', dpi=300)
    log("  ✓ rank.png")
    plt.close()
    
    # Plot 4: Comprehensive summary
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(qubits, lret_times, 'o-', label='LRET', linewidth=2)
    ax1.plot(qubits, cirq_times, 's-', label='Cirq', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Qubits')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(qubits, speedups, color='green', alpha=0.7)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Qubits')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('LRET Speedup')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(qubits, ranks, 'o-', color='purple', linewidth=2)
    ax3.set_xlabel('Qubits')
    ax3.set_ylabel('Final Rank')
    ax3.set_title('Rank Evolution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    memory_full = [(2**q)**2 * 16 / 1e6 for q in qubits]
    memory_lret = [r * 2**q * 16 / 1e6 for r, q in zip(ranks, qubits)]
    ax4.plot(qubits, memory_full, 's-', label='Full DM', linewidth=2)
    ax4.plot(qubits, memory_lret, 'o-', label='LRET', linewidth=2)
    ax4.set_yscale('log')
    ax4.set_xlabel('Qubits')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('Memory Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'LRET vs Cirq Comprehensive Benchmark\n' +
                 f'depth={CONFIG["depth"]}, noise={CONFIG["noise_prob"]*100:.2f}%, ε={CONFIG["epsilon"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'comprehensive_summary.png', dpi=300)
    log("  ✓ comprehensive_summary.png")
    plt.close()

def generate_report(results):
    """Generate markdown report."""
    log("\nGenerating report...")
    
    results = [r for r in results if r is not None]
    
    report = f"""# LRET Automated Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Qubits:** {CONFIG['qubits']}
- **Depth:** {CONFIG['depth']}
- **Noise:** {CONFIG['noise_prob']*100:.4f}% depolarizing per gate
- **Epsilon:** {CONFIG['epsilon']}
- **Trials per config:** {CONFIG['n_trials']}
- **Timeout:** {CONFIG['timeout']}s

## Results

| Qubits | LRET (ms) | Cirq (ms) | Speedup | Rank | Status |
|--------|-----------|-----------|---------|------|--------|
"""
    
    for r in results:
        if r['cirq_mean']:
            report += f"| {r['n_qubits']} | {r['lret_mean']:.1f}±{r['lret_std']:.1f} | " \
                     f"{r['cirq_mean']:.1f}±{r['cirq_std']:.1f} | {r['speedup']:.2f}× | " \
                     f"{r['lret_rank']} | {r['lret_status']} |\n"
        else:
            report += f"| {r['n_qubits']} | {r['lret_mean']:.1f}±{r['lret_std']:.1f} | " \
                     f"FAILED | - | {r['lret_rank']} | {r['lret_status']} |\n"
    
    successful = [r for r in results if r['cirq_mean'] is not None]
    if successful:
        avg_speedup = np.mean([r['speedup'] for r in successful])
        report += f"\n**Average Speedup:** {avg_speedup:.2f}×\n"
    
    report += f"""
## Plots

- `time_comparison.png` - Execution time comparison
- `speedup.png` - Speedup factor
- `rank.png` - Rank evolution
- `comprehensive_summary.png` - All metrics in 2×2 grid

## Raw Data

See `results.json` for complete raw data.
"""
    
    with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
        f.write(report)
    
    log("  ✓ REPORT.md")

def main():
    """Main benchmark execution."""
    log("="*70)
    log("LRET AUTOMATED BENCHMARK")
    log("="*70)
    log(f"Configuration: {CONFIG['qubits']} qubits, depth={CONFIG['depth']}, "
        f"noise={CONFIG['noise_prob']*100:.4f}%, ε={CONFIG['epsilon']}")
    log(f"Output directory: {OUTPUT_DIR}")
    
    # Verify quantum_sim.exe exists
    if not QUANTUM_SIM.exists():
        log(f"ERROR: quantum_sim.exe not found at {QUANTUM_SIM}", "ERROR")
        log("Please run setup script first: 01_setup_environment.ps1", "ERROR")
        sys.exit(1)
    
    # Run benchmarks
    results = []
    for n_qubits in CONFIG['qubits']:
        result = benchmark_single(
            n_qubits=n_qubits,
            depth=CONFIG['depth'],
            noise_prob=CONFIG['noise_prob'],
            epsilon=CONFIG['epsilon'],
            n_trials=CONFIG['n_trials'],
            timeout=CONFIG['timeout']
        )
        results.append(result)
        
        # Save intermediate results
        with open(OUTPUT_DIR / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Generate visualizations
    generate_plots(results)
    
    # Generate report
    generate_report(results)
    
    log("\n" + "="*70)
    log("BENCHMARK COMPLETE!")
    log("="*70)
    log(f"Results saved to: {OUTPUT_DIR}")
    log("")

if __name__ == "__main__":
    main()
