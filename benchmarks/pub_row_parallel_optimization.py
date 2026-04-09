"""
Publication Benchmark 2d: Row-Parallel Optimization Comparison
Compares baseline vs optimized LRET modes using the C++ backend with real
parallel execution via OpenMP, or falls back to a numpy single-thread
baseline measurement (no synthetic speedup injection).

Uses the correct LRET algorithm from:
  Chen, Farquhar, Parrish. npj Quantum Information 7, 61 (2021).

Usage:
  python benchmarks/pub_row_parallel_optimization.py [--quick] [--output-dir results/]

Best branch: row-parallelism-optimization (has C++ parallel modes)
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
    run_lret_simulation,
)

try:
    from python.benchmarks.pub_style import apply_pub_style, save_figure, COLORS, FIGSIZE
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
QUBIT_RANGE_FULL  = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
QUBIT_RANGE_QUICK = [4, 6, 8]
THREAD_COUNTS     = [1, 2, 4, 8, 16]
CIRCUIT_DEPTH     = 13
NOISE_PROB        = 0.001
EPSILON           = 1e-4
N_TRIALS_FULL     = 3
N_TRIALS_QUICK    = 2
TIMEOUT_SECONDS   = 7200.0  # 2 hours per trial

# ──────────────────────────────────────────────────────────────
# C++ backend detection
# ──────────────────────────────────────────────────────────────

def _try_cpp_backend():
    """Check if the C++ LRET backend is available."""
    try:
        from python.qlret.api import simulate_json
        # Quick sanity check
        test = {
            "circuit": {
                "num_qubits": 2,
                "operations": [{"name": "H", "wires": [0]}],
            },
            "config": {"batch_size": 1, "do_truncation": False},
        }
        result = simulate_json(test, use_native=True)
        return result.get('status') == 'success'
    except Exception:
        return False

CPP_AVAILABLE = _try_cpp_backend()


# ──────────────────────────────────────────────────────────────
# Benchmark using numpy (single-thread only, no synthetic speedup)
# ──────────────────────────────────────────────────────────────

def run_numpy_benchmark(n_qubits, depth, noise_prob, n_trials=2, circuit_seed=42):
    """Run the correct LRET algorithm in numpy (single-threaded baseline).

    No synthetic speedup injection — reports real measured wall-clock time.
    """
    times_ms, ranks = [], []

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit = build_random_dense_circuit(n_qubits, depth, rng)

        L, elapsed_ms, max_rank = run_lret_simulation(
            circuit, n_qubits, noise_prob, epsilon=EPSILON
        )
        times_ms.append(elapsed_ms)
        ranks.append(L.shape[1])

    return {
        'mean_ms':    float(np.mean(times_ms)),
        'std_ms':     float(np.std(times_ms)),
        'final_rank': float(np.mean(ranks)),
        'peak_mb':    (2 ** n_qubits) * int(np.mean(ranks)) * 16 / 1e6,
    }


# ──────────────────────────────────────────────────────────────
# Benchmark using C++ backend with real thread control
# ──────────────────────────────────────────────────────────────

def _build_native_circuit_json(n_qubits, depth, noise_prob, rng):
    """Build a random dense circuit using only native C++ backend gate names.

    Same structural form as `_lret_core.build_random_dense_circuit`:
    one layer of single-qubit rotations per qubit, plus alternating-parity
    nearest-neighbor CNOT/CZ entanglers, with depolarizing noise after each
    layer. This mirrors the numpy baseline closely enough to be a fair
    proxy for measuring parallel scaling.
    """
    ops = []
    for d in range(depth):
        # Single-qubit gates
        for q in range(n_qubits):
            angle = float(rng.uniform(0, 2 * np.pi))
            gate = rng.choice(['RX', 'RY', 'RZ', 'H'])
            if gate == 'H':
                ops.append({"name": "H", "wires": [int(q)]})
            else:
                ops.append({"name": gate, "wires": [int(q)], "params": [angle]})
        # Two-qubit entanglers
        start = d % 2
        for i in range(start, n_qubits - 1, 2):
            name = 'CNOT' if rng.uniform() < 0.5 else 'CZ'
            ops.append({"name": name, "wires": [int(i), int(i + 1)]})
        # Noise after each layer
        if noise_prob > 0:
            for q in range(n_qubits):
                ops.append({"name": "DEPOLARIZE", "wires": [int(q)],
                            "params": [float(noise_prob)]})
    return ops


def run_cpp_benchmark(n_qubits, depth, noise_prob, n_threads, mode_config,
                      n_trials=2, circuit_seed=42):
    """Run LRET via C++ backend with actual OpenMP thread parallelism.

    Sets OMP_NUM_THREADS to control real parallel execution.
    """
    from python.qlret.api import simulate_json

    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    times_ms, ranks = [], []

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        ops = _build_native_circuit_json(n_qubits, depth, noise_prob, rng)

        circuit_json = {
            "circuit": {
                "num_qubits": n_qubits,
                "operations": ops,
            },
            "config": {
                "batch_size": 64,
                "do_truncation": True,
                "truncation_threshold": EPSILON,
                **mode_config,
            },
        }

        t0 = time.perf_counter()
        try:
            result = simulate_json(circuit_json, use_native=True,
                                   timeout=TIMEOUT_SECONDS)
        except MemoryError:
            return {'oom': True, 'oom_reason': 'C++ MemoryError'}
        except Exception as exc:
            return {'oom': True, 'oom_reason': f'C++ error: {exc}'}
        elapsed_ms = (time.perf_counter() - t0) * 1000

        times_ms.append(elapsed_ms)
        ranks.append(result.get('final_rank', 1))

    return {
        'mean_ms':    float(np.mean(times_ms)),
        'std_ms':     float(np.std(times_ms)),
        'final_rank': float(np.mean(ranks)),
        'peak_mb':    (2 ** n_qubits) * int(np.mean(ranks)) * 16 / 1e6,
        'oom':        False,
    }


# ──────────────────────────────────────────────────────────────
# Amdahl fitting (applied to REAL measured data only)
# ──────────────────────────────────────────────────────────────

def fit_amdahl(thread_counts, speedups):
    """Fit S(n) = 1/((1-p)+p/n). Returns (p_fit, S_max, n_fine, fit_vals)."""
    def amdahl(n, p):
        return 1.0 / ((1.0 - p) + p / np.asarray(n, dtype=float))

    p_fit = 0.85
    if SCIPY_AVAILABLE:
        try:
            popt, _ = curve_fit(amdahl, thread_counts, speedups,
                                p0=[0.85], bounds=(0.01, 0.99))
            p_fit = float(popt[0])
        except Exception:
            pass

    n_fine = np.linspace(1, max(thread_counts) * 1.2, 100)
    fit = 1.0 / ((1.0 - p_fit) + p_fit / n_fine)
    s_max = 1.0 / (1.0 - p_fit)
    return p_fit, s_max, n_fine.tolist(), fit.tolist()


# ──────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────

def _plot_numpy_only(rows, qubit_range, fig_base):
    """Plot for numpy-only mode (single thread, no parallel comparison)."""
    apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    ns = sorted(set(r['n_qubits'] for r in rows))
    times = [next(r['mean_ms'] for r in rows if r['n_qubits'] == n) for n in ns]
    stds = [next(r['std_ms'] for r in rows if r['n_qubits'] == n) for n in ns]
    ranks = [next(r['final_rank'] for r in rows if r['n_qubits'] == n) for n in ns]
    mems = [next(r['peak_memory_mb'] for r in rows if r['n_qubits'] == n) for n in ns]

    # (a) Time vs qubits
    ax = axes[0]
    ax.semilogy(ns, times, 'o-', color=COLORS['lret'], lw=1.5, ms=5)
    ax.fill_between(ns,
                    [max(0.01, t - s) for t, s in zip(times, stds)],
                    [t + s for t, s in zip(times, stds)],
                    alpha=0.2, color=COLORS['lret'])
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('LRET time (ms)')
    ax.set_title('(a) LRET Baseline Time (numpy, 1 thread)')

    # (b) Rank vs qubits
    ax = axes[1]
    ax.plot(ns, ranks, 'o-', color=COLORS['lret'], lw=1.5, ms=5)
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Final rank $r$')
    ax.set_title('(b) Rank After Truncation')

    # (c) Memory vs qubits
    ax = axes[2]
    ax.semilogy(ns, [max(0.01, m) for m in mems], 'o-', color=COLORS['lret'],
                lw=1.5, ms=5, label='LRET')
    fdm_mems = [(4 ** n) * 16 / 1e6 for n in ns]
    ax.semilogy(ns, fdm_mems, 's--', color=COLORS['cirq_fdm'], lw=1.5, ms=5,
                label='FDM theoretical')
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Peak memory (MB)')
    ax.legend(fontsize=7)
    ax.set_title('(c) Memory Comparison')

    fig.suptitle('LRET Baseline (C++ backend not available — numpy only)',
                 fontsize=10, fontweight='bold')
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
    csv_path = os.path.join(output_dir, f'row_parallel_{datestamp}.csv')
    fig_base = os.path.join(output_dir, f'row_parallel_{datestamp}')

    print(f"\n[2d] Row-Parallel Optimisation — qubits={qubit_range}, "
          f"depth={CIRCUIT_DEPTH}, noise={NOISE_PROB}, trials={n_trials}")

    if CPP_AVAILABLE:
        print("     C++ backend: AVAILABLE — will measure real parallel execution")
        return _run_with_cpp(qubit_range, n_trials, csv_path, fig_base, output_dir)
    else:
        print("     C++ backend: NOT AVAILABLE")
        print("     Running numpy single-thread baseline only (no synthetic speedup)")
        print("     To measure real parallelism, build the C++ backend and re-run.")
        return _run_numpy_only(qubit_range, n_trials, csv_path, fig_base)


def _run_numpy_only(qubit_range, n_trials, csv_path, fig_base):
    """Numpy-only fallback: single-thread baseline, no fake parallelism."""
    all_rows = []

    for n in qubit_range:
        print(f"  n={n} (numpy, 1 thread)...", end='', flush=True)
        r = run_numpy_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB, n_trials=n_trials)

        row = {
            'mode':               'numpy_baseline',
            'n_qubits':           n,
            'n_threads':          1,
            'mean_ms':            r['mean_ms'],
            'std_ms':             r['std_ms'],
            'speedup_vs_baseline': 1.0,
            'peak_memory_mb':     r['peak_mb'],
            'final_rank':         r['final_rank'],
        }
        all_rows.append(row)
        print(f" {r['mean_ms']:.1f} ms, rank={r['final_rank']:.0f}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV: {csv_path}")

    if MATPLOTLIB_AVAILABLE:
        _plot_numpy_only(all_rows, qubit_range, fig_base)

    return all_rows


CSV_FIELDS_CPP = [
    'mode', 'n_qubits', 'n_threads', 'mean_ms', 'std_ms',
    'speedup_vs_baseline', 'peak_memory_mb', 'final_rank', 'status',
]


def _write_csv(csv_path, rows, fields):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fields})


def _run_with_cpp(qubit_range, n_trials, csv_path, fig_base, output_dir):
    """Run with C++ backend: real parallel modes and thread counts.

    Baseline = C++ baseline mode at 1 thread (numpy is too slow at N>=14 to
    serve as a meaningful denominator). All speedups are vs that baseline.
    Per-row intermediate CSV save. If a run OOMs we skip remaining configs at
    that N and continue to smaller-cost work.
    """
    MODES = {
        'baseline': ({}, 'Baseline'),
        'iterative_compression': ({'use_iterative_compression': True},
                                  'Iterative Compression'),
        'full_opt': ({'use_iterative_compression': True,
                      'use_cp_decomposition': True,
                      'use_morton_order': True}, 'All Optimisations'),
    }

    all_rows = []
    baseline_cache = {}  # n_qubits → baseline 1-thread mean_ms

    print(f"  Building per-N C++ baseline (mode=baseline, threads=1)...")
    for n in qubit_range:
        print(f"    n={n:2d} baseline...", end='', flush=True)
        r = run_cpp_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB, 1, {},
                              n_trials=1, circuit_seed=42)
        if r.get('oom'):
            print(f" OOM ({r.get('oom_reason','')}) — stopping at this N")
            baseline_cache[n] = float('nan')
            # Stop adding qubits — anything bigger will also OOM
            qubit_range = [m for m in qubit_range if m <= n]
            break
        baseline_cache[n] = r['mean_ms']
        print(f" {r['mean_ms']:.1f} ms (rank={r['final_rank']:.0f})")

    print(f"\n  Sweeping {len(qubit_range)} qubit counts × {len(MODES)} modes "
          f"× {len(THREAD_COUNTS)} thread counts")
    for n in qubit_range:
        if not np.isfinite(baseline_cache.get(n, float('nan'))):
            continue
        for mode_key, (mode_config, mode_label) in MODES.items():
            for n_threads in THREAD_COUNTS:
                print(f"  n={n:2d} mode={mode_key:22s} t={n_threads:2d}...",
                      end='', flush=True)

                r = run_cpp_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB, n_threads,
                                      mode_config, n_trials=n_trials)

                if r.get('oom'):
                    row = {
                        'mode': mode_key, 'n_qubits': n, 'n_threads': n_threads,
                        'mean_ms': float('nan'), 'std_ms': float('nan'),
                        'speedup_vs_baseline': float('nan'),
                        'peak_memory_mb': float('nan'),
                        'final_rank': float('nan'),
                        'status': f"oom:{r.get('oom_reason','')}",
                    }
                    all_rows.append(row)
                    _write_csv(csv_path, all_rows, CSV_FIELDS_CPP)
                    print(f" OOM ({r.get('oom_reason','')})")
                    continue

                speedup = baseline_cache[n] / max(r['mean_ms'], 1e-6)
                row = {
                    'mode':               mode_key,
                    'n_qubits':           n,
                    'n_threads':          n_threads,
                    'mean_ms':            r['mean_ms'],
                    'std_ms':             r['std_ms'],
                    'speedup_vs_baseline': speedup,
                    'peak_memory_mb':     r['peak_mb'],
                    'final_rank':         r['final_rank'],
                    'status':             'ok',
                }
                all_rows.append(row)
                _write_csv(csv_path, all_rows, CSV_FIELDS_CPP)
                print(f" {r['mean_ms']:8.1f} ms (x{speedup:.2f})")

    print(f"\n  Final CSV: {csv_path}  ({len(all_rows)} rows)")
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    run(args.output_dir, args.quick)


if __name__ == '__main__':
    main()
