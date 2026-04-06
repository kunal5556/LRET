"""
Publication Benchmark 2a: LRET vs Cirq/Qiskit Density-Matrix Simulators
Generates IEEE double-column 4-panel figure comparing timing, speedup, distortion, rank.

Uses the correct LRET algorithm from:
  Chen, Farquhar, Parrish. "Low-rank density-matrix evolution for noisy quantum circuits."
  npj Quantum Information 7, 61 (2021).

Both LRET and competitors run the SAME dense random circuit with the SAME
depolarizing noise model, ensuring an apple-to-apple comparison.

Usage:
  python benchmarks/pub_lret_vs_cirq.py [--quick] [--output-dir results/]
"""

import os
import sys
import csv
import time
import argparse
import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from _lret_core import (
    build_random_dense_circuit,
    build_cirq_circuit_from_layers,
    run_lret_simulation,
    reconstruct_density_matrix,
    trace_distance,
    compute_distortion,
    compute_probability_distribution,
    probability_tvd,
)

try:
    from python.benchmarks.pub_style import apply_pub_style, save_figure, COLORS, FIGSIZE
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# Configuration — matches paper parameters
# ──────────────────────────────────────────────────────────────
QUBIT_RANGE_FULL  = [4, 6, 8, 10, 12]
QUBIT_RANGE_QUICK = [4, 6, 8]
CIRCUIT_DEPTH     = 13          # Paper uses D=13 for benchmarks
NOISE_PROB        = 0.001       # Paper: p = 0.1%
EPSILON           = 1e-4        # Paper: epsilon = 10^-4
N_TRIALS_FULL     = 3
N_TRIALS_QUICK    = 2

# ──────────────────────────────────────────────────────────────
# LRET benchmark (correct algorithm)
# ──────────────────────────────────────────────────────────────

def run_lret_benchmark(n_qubits, depth, noise_prob, epsilon=EPSILON,
                       n_trials=3, circuit_seed=42):
    """Run LRET simulation using the paper's algorithm.

    - Dense random circuit with 1q + 2q gates
    - Proper Kraus channel noise (rank expansion by factor 4 per qubit)
    - Per-qubit iterative truncation (paper Section III.B)
    """
    times_ms = []
    final_ranks = []
    max_ranks = []
    L_last = None

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit = build_random_dense_circuit(n_qubits, depth, rng)

        L, elapsed_ms, max_rank_seen = run_lret_simulation(
            circuit, n_qubits, noise_prob, epsilon=epsilon
        )

        times_ms.append(elapsed_ms)
        final_ranks.append(L.shape[1])
        max_ranks.append(max_rank_seen)
        L_last = L

    return {
        'n_qubits':       n_qubits,
        'mean_ms':        float(np.mean(times_ms)),
        'std_ms':         float(np.std(times_ms)),
        'final_rank':     float(np.mean(final_ranks)),
        'max_rank':       float(np.mean(max_ranks)),
        'dim':            2 ** n_qubits,
        'L_final':        L_last,
        'circuit_seed':   circuit_seed,
    }


# ──────────────────────────────────────────────────────────────
# Cirq benchmark (same circuit, full density matrix)
# ──────────────────────────────────────────────────────────────

def run_cirq_benchmark(n_qubits, depth, noise_prob, n_trials=3, circuit_seed=42):
    """Run Cirq DensityMatrixSimulator on the same circuit as LRET."""
    try:
        import cirq
    except ImportError:
        return {
            'n_qubits': n_qubits, 'mean_ms': float('nan'),
            'std_ms': float('nan'), 'estimated': True,
        }

    times_ms = []
    rho_last = None

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit_layers = build_random_dense_circuit(n_qubits, depth, rng)
        cirq_circuit = build_cirq_circuit_from_layers(circuit_layers, n_qubits, noise_prob)

        sim = cirq.DensityMatrixSimulator()
        t0 = time.perf_counter()
        try:
            result = sim.simulate(cirq_circuit)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed_ms)
            rho_last = result.final_density_matrix
        except MemoryError:
            return {
                'n_qubits': n_qubits, 'mean_ms': float('nan'),
                'std_ms': float('nan'), 'oom': True,
            }

    return {
        'n_qubits': n_qubits,
        'mean_ms':  float(np.mean(times_ms)),
        'std_ms':   float(np.std(times_ms)),
        'oom':      False,
        'rho':      rho_last,
    }


# ──────────────────────────────────────────────────────────────
# Error computation (Paper Eq. 10)
# ──────────────────────────────────────────────────────────────

def _compute_error_metrics(L_lret, rho_cirq, n_qubits, depth, noise_prob, circuit_seed):
    """Compute distortion and probability TVD.

    Distortion = T(rho_LRET, rho_exact) / T(rho_exact, rho_noiseless)
    where rho_exact = Cirq result (full density matrix with noise),
          rho_noiseless = Cirq result without noise.
    """
    if L_lret is None or rho_cirq is None:
        return float('nan'), float('nan')

    try:
        import cirq

        # Get noiseless reference
        rng = np.random.default_rng(circuit_seed)
        circuit_layers = build_random_dense_circuit(n_qubits, depth, rng)
        noiseless_circuit = build_cirq_circuit_from_layers(circuit_layers, n_qubits, 0.0)
        sim = cirq.DensityMatrixSimulator()
        result_noiseless = sim.simulate(noiseless_circuit)
        rho_noiseless = result_noiseless.final_density_matrix

        # Compute distortion
        distortion = compute_distortion(L_lret, rho_cirq, rho_noiseless)

        # Probability TVD
        prob_lret = compute_probability_distribution(L_lret)
        prob_exact = np.diag(rho_cirq).real
        tvd = probability_tvd(prob_lret, prob_exact)

        return distortion, tvd
    except Exception:
        return float('nan'), float('nan')


# ──────────────────────────────────────────────────────────────
# Main benchmark runner
# ──────────────────────────────────────────────────────────────

def run(output_dir='results', quick=False):
    qubit_range = QUBIT_RANGE_QUICK if quick else QUBIT_RANGE_FULL
    n_trials = N_TRIALS_QUICK if quick else N_TRIALS_FULL

    os.makedirs(output_dir, exist_ok=True)
    datestamp = datetime.datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(output_dir, f'lret_vs_cirq_{datestamp}.csv')
    fig_base = os.path.join(output_dir, f'lret_vs_cirq_{datestamp}')

    print(f"\n[2a] LRET vs Cirq — qubits={qubit_range}, depth={CIRCUIT_DEPTH}, "
          f"noise={NOISE_PROB}, eps={EPSILON}, trials={n_trials}")
    print(f"     Circuit: dense random (1q+2q gates), depolarizing Kraus noise")

    rows = []
    for n in qubit_range:
        print(f"  n={n}...", end='', flush=True)

        seed = 42
        lret = run_lret_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB, EPSILON,
                                  n_trials=n_trials, circuit_seed=seed)
        cirq_r = run_cirq_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB,
                                    n_trials=n_trials, circuit_seed=seed)

        lret_ms = lret['mean_ms']
        cirq_ms = cirq_r.get('mean_ms', float('nan'))

        speedup = cirq_ms / lret_ms if np.isfinite(cirq_ms) and lret_ms > 0 else float('nan')

        # Compute distortion (only feasible for small N)
        distortion, tvd = float('nan'), float('nan')
        if n <= 12:
            distortion, tvd = _compute_error_metrics(
                lret.get('L_final'), cirq_r.get('rho'), n, CIRCUIT_DEPTH, NOISE_PROB, seed
            )

        row = {
            'n_qubits':     n,
            'lret_mean_ms': lret_ms,
            'lret_std_ms':  lret['std_ms'],
            'cirq_mean_ms': cirq_ms,
            'cirq_std_ms':  cirq_r.get('std_ms', 0.0),
            'speedup':      speedup,
            'distortion':   distortion,
            'prob_tvd':     tvd,
            'final_rank':   lret['final_rank'],
            'max_rank':     lret['max_rank'],
            'hilbert_dim':  2 ** n,
        }
        rows.append(row)
        rank_pct = 100 * lret['final_rank'] / (2 ** n)
        print(f" LRET={lret_ms:.1f}ms, Cirq={cirq_ms:.1f}ms, "
              f"speedup={speedup:.2f}x, rank={lret['final_rank']:.0f} "
              f"({rank_pct:.1f}% of 2^n), distortion={distortion:.4f}")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    if MATPLOTLIB_AVAILABLE:
        _plot(rows, fig_base)

    return rows


def _plot(rows, fig_base):
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['double_col'])

    ns = [r['n_qubits'] for r in rows]
    lret_mean = [r['lret_mean_ms'] for r in rows]
    lret_std = [r['lret_std_ms'] for r in rows]
    cirq_mean = [r['cirq_mean_ms'] for r in rows]
    speedup = [r['speedup'] for r in rows]
    distortion = [r['distortion'] for r in rows]
    rank = [r['final_rank'] for r in rows]
    rank_pct = [100 * r['final_rank'] / r['hilbert_dim'] for r in rows]

    # [0,0] Time vs qubits
    ax = axes[0, 0]
    ax.semilogy(ns, lret_mean, 'o-', color=COLORS['lret'], label='LRET')
    ax.fill_between(ns,
                    [max(1e-3, m - s) for m, s in zip(lret_mean, lret_std)],
                    [m + s for m, s in zip(lret_mean, lret_std)],
                    alpha=0.2, color=COLORS['lret'])
    valid_cirq = [(n, t) for n, t in zip(ns, cirq_mean) if np.isfinite(t)]
    if valid_cirq:
        ax.semilogy(*zip(*valid_cirq), 's--', color=COLORS['cirq_fdm'],
                    label='Cirq (DensityMatrix)')
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Simulation time (ms)')
    ax.legend(framealpha=0.9)
    ax.set_title('(a) Simulation Time')

    # [0,1] Speedup
    ax = axes[0, 1]
    valid_s = [(n, s) for n, s in zip(ns, speedup) if np.isfinite(s)]
    if valid_s:
        ns_v, sp_v = zip(*valid_s)
        ax.plot(ns_v, sp_v, 'o-', color=COLORS['cirq_fdm'], label='vs Cirq')
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1.0, label='LRET = Cirq')
    ax.fill_between(ns,
                    [1] * len(ns),
                    [max(1, s) if np.isfinite(s) else 1 for s in speedup],
                    alpha=0.15, color=COLORS['lret'], label='LRET wins')
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Speedup (x)')
    ax.legend(framealpha=0.9)
    ax.set_title('(b) Speedup Ratio')

    # [1,0] Distortion (paper Eq. 10)
    ax = axes[1, 0]
    valid_d = [(n, d) for n, d in zip(ns, distortion) if np.isfinite(d)]
    if valid_d:
        ax.plot(*zip(*valid_d), 'o-', color=COLORS['lret'])
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Distortion (paper Eq. 10)')
    ax.set_title(f'(c) Approximation Error ($p={NOISE_PROB}$, $\\epsilon={EPSILON}$)')

    # [1,1] LRET final rank
    ax = axes[1, 1]
    ax.plot(ns, rank, 'o-', color=COLORS['lret'], label='Final rank $r$')
    ax2 = ax.twinx()
    ax2.plot(ns, rank_pct, 's--', color=COLORS.get('lret_opt', 'green'),
             alpha=0.6, label='% of $2^n$')
    ax2.set_ylabel('% of Hilbert space', color=COLORS.get('lret_opt', 'green'))
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Final LRET rank $r$')
    ax.set_title('(d) Rank Compression')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, framealpha=0.9)

    fig.suptitle('LRET vs Cirq: Same Circuit, Same Noise', fontsize=11, fontweight='bold')
    save_figure(fig, fig_base)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='LRET vs Cirq benchmark (correct algorithm)')
    parser.add_argument('--quick', action='store_true', help='Reduced qubit range / trials')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()
    run(args.output_dir, args.quick)


if __name__ == '__main__':
    main()
