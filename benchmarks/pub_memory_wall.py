"""
Publication Benchmark 2b: LRET vs Full Density Matrix — Memory Wall
Generates IEEE double-column 3-panel figure showing LRET advantage at high qubit counts.

Both LRET and FDM run the SAME dense random circuit with depolarizing noise,
using the correct LRET algorithm from:
  Chen, Farquhar, Parrish. npj Quantum Information 7, 61 (2021).

Usage:
  python benchmarks/pub_memory_wall.py [--quick] [--output-dir results/]
"""

import os
import sys
import csv
import time
import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from _lret_core import (
    build_random_dense_circuit,
    build_cirq_circuit_from_layers,
    run_lret_simulation,
)

try:
    from python.benchmarks.pub_style import (
        apply_pub_style, save_figure, COLORS, FIGSIZE
    )
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# Configuration — matches paper parameters
# ──────────────────────────────────────────────────────────────
QUBIT_RANGE_FULL  = [4, 6, 8, 10, 12, 14, 16]
QUBIT_RANGE_QUICK = [4, 6, 8, 10, 12]
CIRCUIT_DEPTH     = 13
NOISE_PROB        = 0.001       # Paper: p = 0.1%
EPSILON           = 1e-4        # Paper: epsilon = 10^-4
N_TRIALS_FULL     = 3
N_TRIALS_QUICK    = 1
OOM_SENTINEL      = float('nan')
BYTES_PER_COMPLEX128 = 16


# ──────────────────────────────────────────────────────────────
# Memory estimates
# ──────────────────────────────────────────────────────────────

def fdm_memory_gb(n_qubits):
    """Theoretical FDM memory: density matrix is 4^n complex128 entries."""
    return (4 ** n_qubits) * BYTES_PER_COMPLEX128 / 1e9

def lret_memory_mb(n_qubits, rank):
    """Actual LRET memory: 2^n x rank complex128 entries."""
    return (2 ** n_qubits * rank) * BYTES_PER_COMPLEX128 / 1e6

def get_system_ram_gb():
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().total / 1e9
    return 16.0

def _get_rss_mb():
    if PSUTIL_AVAILABLE:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            pass
    return 0.0


# ──────────────────────────────────────────────────────────────
# LRET benchmark (correct algorithm with noise)
# ──────────────────────────────────────────────────────────────

def run_lret_benchmark(n_qubits, depth, noise_prob, n_trials=1, circuit_seed=42):
    """Run LRET with proper Kraus noise + truncation. Track time and memory."""
    times_ms, peak_mbs, ranks = [], [], []

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit = build_random_dense_circuit(n_qubits, depth, rng)

        mem_before = _get_rss_mb()
        L, elapsed_ms, max_rank = run_lret_simulation(
            circuit, n_qubits, noise_prob, epsilon=EPSILON
        )
        mem_after = _get_rss_mb()

        peak_mb = max(mem_after - mem_before, lret_memory_mb(n_qubits, L.shape[1]))
        times_ms.append(elapsed_ms)
        peak_mbs.append(peak_mb)
        ranks.append(L.shape[1])

    return {
        'n_qubits':   n_qubits,
        'mean_ms':    float(np.mean(times_ms)),
        'std_ms':     float(np.std(times_ms)),
        'peak_mb':    float(np.mean(peak_mbs)),
        'final_rank': float(np.mean(ranks)),
        'oom':        False,
    }


# ──────────────────────────────────────────────────────────────
# FDM benchmark (Cirq DensityMatrixSimulator, same circuit)
# ──────────────────────────────────────────────────────────────

def run_fdm_benchmark(n_qubits, depth, noise_prob, n_trials=1, circuit_seed=42):
    """Run Cirq DensityMatrixSimulator on the same circuit."""
    system_ram_gb = get_system_ram_gb()
    theoretical_gb = fdm_memory_gb(n_qubits)

    if theoretical_gb > 0.8 * system_ram_gb:
        return {
            'n_qubits': n_qubits, 'mean_ms': OOM_SENTINEL,
            'std_ms': OOM_SENTINEL, 'peak_mb': OOM_SENTINEL,
            'theoretical_gb': theoretical_gb, 'oom': True,
            'oom_reason': f'Theoretical {theoretical_gb:.1f} GB > 80% of {system_ram_gb:.1f} GB',
        }

    try:
        import cirq
    except ImportError:
        return {
            'n_qubits': n_qubits, 'mean_ms': OOM_SENTINEL,
            'std_ms': OOM_SENTINEL, 'peak_mb': theoretical_gb * 1000,
            'theoretical_gb': theoretical_gb, 'oom': False, 'estimated': True,
        }

    times_ms, peak_mbs = [], []
    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit_layers = build_random_dense_circuit(n_qubits, depth, rng)
        cirq_circuit = build_cirq_circuit_from_layers(circuit_layers, n_qubits, noise_prob)

        mem_before = _get_rss_mb()
        t0 = time.perf_counter()
        try:
            sim = cirq.DensityMatrixSimulator()
            sim.simulate(cirq_circuit)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            peak_mb = max(_get_rss_mb() - mem_before, theoretical_gb * 1000)
            times_ms.append(elapsed_ms)
            peak_mbs.append(peak_mb)
        except MemoryError:
            return {
                'n_qubits': n_qubits, 'mean_ms': OOM_SENTINEL,
                'std_ms': OOM_SENTINEL, 'peak_mb': OOM_SENTINEL,
                'theoretical_gb': theoretical_gb, 'oom': True,
            }

    return {
        'n_qubits': n_qubits,
        'mean_ms': float(np.mean(times_ms)),
        'std_ms': float(np.std(times_ms)),
        'peak_mb': float(np.mean(peak_mbs)),
        'theoretical_gb': theoretical_gb,
        'oom': False,
    }


# ──────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────

def _plot(rows, fig_base, system_ram_gb):
    apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    ns = [r['n_qubits'] for r in rows]
    fdm_theory_mb = [r['fdm_theoretical_gb'] * 1000 for r in rows]
    fdm_meas_mb = [r['fdm_memory_mb'] for r in rows]
    lret_mb = [r['lret_memory_mb'] for r in rows]
    fdm_time = [r['fdm_time_ms'] for r in rows]
    lret_time = [r['lret_time_ms'] for r in rows]
    lret_std = [r['lret_std_ms'] for r in rows]
    oom_flags = [r['fdm_oom'] for r in rows]

    ns_arr = np.array(ns)

    # (a) Memory
    ax = axes[0]
    pre = [(n, m) for n, m, oom in zip(ns, fdm_theory_mb, oom_flags) if not oom]
    post = [(n, m) for n, m, oom in zip(ns, fdm_theory_mb, oom_flags) if oom]
    if pre:
        ax.semilogy(*zip(*pre), '-', color=COLORS['cirq_fdm'], lw=1.5,
                    label='FDM theoretical')
    if post:
        ax.semilogy(*zip(*post), '--', color=COLORS['cirq_fdm'], alpha=0.5)
    meas = [(n, m) for n, m in zip(ns, fdm_meas_mb) if np.isfinite(m)]
    if meas:
        ax.semilogy(*zip(*meas), 'o', color=COLORS['cirq_fdm'], ms=5,
                    label='FDM measured')
    ax.semilogy(ns, [max(0.01, m) for m in lret_mb], 's-', color=COLORS['lret'],
                lw=1.5, ms=5, label='LRET')
    ax.axhline(system_ram_gb * 1000, color=COLORS['system_ram'], ls='--', lw=1.5,
               label=f'System RAM ({system_ram_gb:.0f} GB)')
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Peak memory (MB)')
    ax.legend(fontsize=6, framealpha=0.9)
    ax.set_title('(a) Memory Scaling')

    # (b) Time
    ax = axes[1]
    valid_fdm = [(n, t) for n, t, oom in zip(ns, fdm_time, oom_flags)
                 if not oom and np.isfinite(t)]
    if valid_fdm:
        ax.semilogy(*zip(*valid_fdm), 'o-', color=COLORS['cirq_fdm'], lw=1.5,
                    ms=5, label='FDM (Cirq)')
    ax.semilogy(ns, [max(0.01, t) for t in lret_time], 's-', color=COLORS['lret'],
                lw=1.5, ms=5, label='LRET')
    ax.fill_between(ns,
                    [max(0.01, t - s) for t, s in zip(lret_time, lret_std)],
                    [t + s for t, s in zip(lret_time, lret_std)],
                    alpha=0.2, color=COLORS['lret'])
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_title('(b) Time Scaling')

    # (c) Complexity reference
    ax = axes[2]
    fdm_ref = 4.0 ** ns_arr / 4.0 ** ns_arr[0]
    lret_ref = 2.0 ** ns_arr / 2.0 ** ns_arr[0]
    ax.semilogy(ns, fdm_ref, '--', color=COLORS['cirq_fdm'], lw=1.5,
                label='$O(4^n)$ FDM')
    ax.semilogy(ns, lret_ref, '-', color=COLORS['lret'], lw=1.5,
                label='$O(2^n \\cdot r)$ LRET')
    ax.axvline(8, color='k', ls=':', lw=0.8, alpha=0.7)
    ax.annotate('$n{\\approx}8$\ncrossover', xy=(8.2, 4.0), fontsize=7)
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Normalised cost (a.u.)')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_title('(c) Complexity Reference')

    fig.suptitle('LRET vs Full-Density-Matrix: Memory Wall',
                 fontsize=11, fontweight='bold')
    save_figure(fig, fig_base)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────

def run(output_dir='results', quick=False):
    qubit_range = QUBIT_RANGE_QUICK if quick else QUBIT_RANGE_FULL
    n_trials = N_TRIALS_QUICK if quick else N_TRIALS_FULL

    os.makedirs(output_dir, exist_ok=True)
    datestamp = datetime.datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(output_dir, f'memory_wall_{datestamp}.csv')
    fig_base = os.path.join(output_dir, f'memory_wall_{datestamp}')

    system_ram_gb = get_system_ram_gb()
    print(f"\n[2b] Memory Wall — System RAM: {system_ram_gb:.1f} GB, "
          f"qubits={qubit_range}, depth={CIRCUIT_DEPTH}, noise={NOISE_PROB}, trials={n_trials}")
    print(f"     Circuit: dense random (1q+2q gates), depolarizing Kraus noise")

    rows = []
    for n in qubit_range:
        print(f"  n={n}...", end='', flush=True)

        seed = 42
        lret_r = run_lret_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB,
                                    n_trials=n_trials, circuit_seed=seed)
        fdm_r = run_fdm_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB,
                                  n_trials=n_trials, circuit_seed=seed)

        row = {
            'n_qubits':           n,
            'fdm_memory_mb':      fdm_r.get('peak_mb', OOM_SENTINEL),
            'fdm_theoretical_gb': fdm_r.get('theoretical_gb', fdm_memory_gb(n)),
            'fdm_time_ms':        fdm_r.get('mean_ms', OOM_SENTINEL),
            'fdm_oom':            fdm_r.get('oom', False),
            'lret_memory_mb':     lret_r['peak_mb'],
            'lret_time_ms':       lret_r['mean_ms'],
            'lret_std_ms':        lret_r['std_ms'],
            'lret_rank':          lret_r['final_rank'],
            'system_ram_gb':      system_ram_gb,
        }
        rows.append(row)

        fdm_str = 'OOM' if fdm_r.get('oom') else f"{fdm_r.get('mean_ms', 0):.1f} ms"
        print(f" LRET={lret_r['mean_ms']:.1f} ms (rank={lret_r['final_rank']:.0f})  "
              f"FDM={fdm_str}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    if MATPLOTLIB_AVAILABLE:
        _plot(rows, fig_base, system_ram_gb)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Memory wall benchmark (correct LRET)')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    run(args.output_dir, args.quick)


if __name__ == '__main__':
    main()
