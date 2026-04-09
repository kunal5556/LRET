"""
Publication Benchmark 2a: LRET vs Cirq Density-Matrix Simulator (Round 2)
Generates IEEE double-column 4-panel figure comparing timing, speedup, fidelity/
distortion, and rank.

Round 2 changes (vs Round 1):
  - LRET uses the C++ quantum_sim backend (10-100x faster than the numpy reference)
  - Quantum fidelity F(rho_LRET, rho_Cirq) is now computed and reported
  - Trace and purity sanity diagnostics on every row
  - Per-row intermediate CSV save (crash-safe)
  - Cirq is only run when its dense density matrix fits in RAM
  - Qubit range extended to N=20 with graceful OOM handling
  - Seed-bug fix: error metrics use the SAME seed/circuit as the timed run

Both LRET and Cirq run the SAME dense random circuit with the SAME depolarizing
noise model. The first trial's circuit is also the one used for error metrics,
so distortion/fidelity correspond exactly to the L matrix that was timed.

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
    run_lret_cpp,
    reconstruct_density_matrix,
    compute_distortion,
    compute_fidelity,
    compute_probability_distribution,
    compute_purity_lret,
    compute_trace_lret,
    probability_tvd,
    trace_distance,
)

try:
    from python.benchmarks.pub_style import apply_pub_style, save_figure, COLORS, FIGSIZE
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import psutil
    _SYSTEM_RAM_GB = psutil.virtual_memory().total / 1e9
except ImportError:
    _SYSTEM_RAM_GB = 16.0

# ──────────────────────────────────────────────────────────────
# Configuration — matches paper parameters, scaled to N=20
# ──────────────────────────────────────────────────────────────
QUBIT_RANGE_FULL  = [4, 6, 8, 10, 12, 14, 16, 18, 20]
QUBIT_RANGE_QUICK = [4, 6, 8]
CIRCUIT_DEPTH     = 13          # Paper uses D=13 for benchmarks
NOISE_PROB        = 0.001       # Paper: p = 0.1%
EPSILON           = 1e-4        # Paper: epsilon = 10^-4
N_TRIALS_FULL     = 3
N_TRIALS_QUICK    = 2

# Cirq's DM is 2^n x 2^n complex128 → 4^n * 16 bytes. Cap at 50% of RAM.
CIRQ_RAM_FRACTION = 0.5
# Use the numpy reference LRET path for tiny systems (faster than spawning a
# subprocess), C++ for everything else.
NUMPY_LRET_MAX_N = 6


def _cirq_dm_bytes(n):
    return 16 * (1 << (2 * n))


def _cirq_feasible(n):
    return _cirq_dm_bytes(n) < CIRQ_RAM_FRACTION * _SYSTEM_RAM_GB * 1e9


# ──────────────────────────────────────────────────────────────
# LRET benchmark: returns timing + L from FIRST trial (same circuit
# as later error-metric calls).
# ──────────────────────────────────────────────────────────────

def run_lret_benchmark(n_qubits, depth, noise_prob, epsilon=EPSILON,
                       n_trials=3, circuit_seed=42):
    use_cpp = n_qubits > NUMPY_LRET_MAX_N
    times_ms = []
    final_ranks = []
    L_first = None

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit = build_random_dense_circuit(n_qubits, depth, rng)

        if use_cpp:
            try:
                L, elapsed_ms, final_rank = run_lret_cpp(
                    circuit, n_qubits, noise_prob, epsilon=epsilon,
                    timeout_s=7200.0, export_state=True,
                )
            except MemoryError:
                return {'n_qubits': n_qubits, 'oom': True}
        else:
            L, elapsed_ms, _max_rank = run_lret_simulation(
                circuit, n_qubits, noise_prob, epsilon=epsilon
            )
            final_rank = L.shape[1]

        times_ms.append(elapsed_ms)
        final_ranks.append(final_rank)
        if trial == 0:
            L_first = L

    return {
        'n_qubits':       n_qubits,
        'mean_ms':        float(np.mean(times_ms)),
        'std_ms':         float(np.std(times_ms)),
        'final_rank':     float(np.mean(final_ranks)),
        'dim':            2 ** n_qubits,
        'L_first':        L_first,
        'circuit_seed':   circuit_seed,
        'backend':        'cpp' if use_cpp else 'numpy',
        'oom':            False,
    }


# ──────────────────────────────────────────────────────────────
# Cirq benchmark: returns timing + rho from FIRST trial (same circuit).
# ──────────────────────────────────────────────────────────────

def run_cirq_benchmark(n_qubits, depth, noise_prob, n_trials=3, circuit_seed=42):
    if not _cirq_feasible(n_qubits):
        return {
            'n_qubits': n_qubits, 'mean_ms': float('nan'),
            'std_ms': float('nan'), 'oom': True,
            'reason': f'DM would need {_cirq_dm_bytes(n_qubits)/1e9:.1f} GB '
                      f'(>{CIRQ_RAM_FRACTION*100:.0f}% of {_SYSTEM_RAM_GB:.0f} GB)'
        }

    try:
        import cirq
    except ImportError:
        return {'n_qubits': n_qubits, 'mean_ms': float('nan'),
                'std_ms': float('nan'), 'oom': True, 'reason': 'cirq not installed'}

    qubits = cirq.LineQubit.range(n_qubits)
    times_ms = []
    rho_first = None

    for trial in range(n_trials):
        rng = np.random.default_rng(circuit_seed + trial)
        circuit_layers = build_random_dense_circuit(n_qubits, depth, rng)
        cirq_circuit = build_cirq_circuit_from_layers(circuit_layers, n_qubits, noise_prob)

        sim = cirq.DensityMatrixSimulator()
        try:
            t0 = time.perf_counter()
            result = sim.simulate(cirq_circuit, qubit_order=qubits)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            return {'n_qubits': n_qubits, 'mean_ms': float('nan'),
                    'std_ms': float('nan'), 'oom': True, 'reason': 'allocation failed'}
        times_ms.append(elapsed_ms)
        if trial == 0:
            rho_first = np.asarray(result.final_density_matrix, dtype=complex)

    return {
        'n_qubits': n_qubits,
        'mean_ms':  float(np.mean(times_ms)),
        'std_ms':   float(np.std(times_ms)),
        'oom':      False,
        'rho':      rho_first,
    }


# ──────────────────────────────────────────────────────────────
# Error metrics: distortion (paper Eq. 10), fidelity, prob TVD
# All use the SAME circuit (circuit_seed, no offset) → same L/rho as the
# timed first trial.
# ──────────────────────────────────────────────────────────────

def _compute_error_metrics(L_lret, rho_cirq, n_qubits, depth, noise_prob, circuit_seed):
    if L_lret is None or rho_cirq is None:
        return {
            'fidelity': float('nan'),
            'distortion': float('nan'),
            'prob_tvd': float('nan'),
            'trace_distance': float('nan'),
        }

    try:
        # Build the noiseless reference using the SAME first-trial seed
        rng = np.random.default_rng(circuit_seed)
        circuit_layers = build_random_dense_circuit(n_qubits, depth, rng)
        import cirq
        qubits = cirq.LineQubit.range(n_qubits)
        noiseless_circuit = build_cirq_circuit_from_layers(circuit_layers, n_qubits, 0.0)
        sim = cirq.DensityMatrixSimulator()
        rho_noiseless = np.asarray(
            sim.simulate(noiseless_circuit, qubit_order=qubits).final_density_matrix,
            dtype=complex,
        )

        rho_cirq_c = np.asarray(rho_cirq, dtype=complex)
        fidelity = compute_fidelity(L_lret, rho_cirq_c)
        distortion = compute_distortion(L_lret, rho_cirq_c, rho_noiseless)
        rho_lret = reconstruct_density_matrix(L_lret)
        td = trace_distance(rho_lret, rho_cirq_c)

        prob_lret = compute_probability_distribution(L_lret)
        prob_exact = np.real(np.diag(rho_cirq_c))
        tvd = probability_tvd(prob_lret, prob_exact)

        return {
            'fidelity': float(fidelity),
            'distortion': float(distortion),
            'prob_tvd': float(tvd),
            'trace_distance': float(td),
        }
    except Exception as exc:
        print(f"      [warn] error-metric computation failed: {exc}")
        return {
            'fidelity': float('nan'),
            'distortion': float('nan'),
            'prob_tvd': float('nan'),
            'trace_distance': float('nan'),
        }


# ──────────────────────────────────────────────────────────────
# Main runner with intermediate save
# ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    'n_qubits', 'lret_backend', 'lret_mean_ms', 'lret_std_ms',
    'cirq_mean_ms', 'cirq_std_ms', 'cirq_status',
    'speedup', 'fidelity', 'distortion', 'trace_distance', 'prob_tvd',
    'lret_trace', 'lret_purity',
    'final_rank', 'rank_pct', 'hilbert_dim',
]


def _write_csv(csv_path, rows):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in CSV_FIELDS})


def run(output_dir='results', quick=False):
    qubit_range = QUBIT_RANGE_QUICK if quick else QUBIT_RANGE_FULL
    n_trials = N_TRIALS_QUICK if quick else N_TRIALS_FULL

    os.makedirs(output_dir, exist_ok=True)
    datestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'lret_vs_cirq_{datestamp}.csv')
    fig_base = os.path.join(output_dir, f'lret_vs_cirq_{datestamp}')

    print(f"\n[2a Round 2] LRET vs Cirq")
    print(f"  qubits={qubit_range}  depth={CIRCUIT_DEPTH}  noise={NOISE_PROB}  "
          f"eps={EPSILON}  trials={n_trials}")
    print(f"  system RAM = {_SYSTEM_RAM_GB:.1f} GB; "
          f"Cirq cap = {CIRQ_RAM_FRACTION*100:.0f}%; "
          f"LRET backend: numpy for N<={NUMPY_LRET_MAX_N}, C++ otherwise")
    print(f"  CSV:   {csv_path}")

    rows = []
    for n in qubit_range:
        print(f"  n={n:2d} ", end='', flush=True)

        seed = 42

        try:
            lret = run_lret_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB, EPSILON,
                                      n_trials=n_trials, circuit_seed=seed)
        except MemoryError:
            print("LRET OOM")
            rows.append({'n_qubits': n, 'cirq_status': 'lret_oom'})
            _write_csv(csv_path, rows)
            break
        except Exception as exc:
            print(f"LRET failed: {exc}")
            rows.append({'n_qubits': n, 'cirq_status': f'lret_err:{exc}'})
            _write_csv(csv_path, rows)
            continue

        if lret.get('oom'):
            print("LRET OOM")
            rows.append({'n_qubits': n, 'cirq_status': 'lret_oom'})
            _write_csv(csv_path, rows)
            break

        cirq_r = run_cirq_benchmark(n, CIRCUIT_DEPTH, NOISE_PROB,
                                    n_trials=n_trials, circuit_seed=seed)

        lret_ms = lret['mean_ms']
        cirq_ms = cirq_r.get('mean_ms', float('nan'))
        speedup = (cirq_ms / lret_ms) if (np.isfinite(cirq_ms) and lret_ms > 0) else float('nan')

        # Sanity diagnostics on the L matrix actually used for error metrics
        L = lret.get('L_first')
        lret_trace = compute_trace_lret(L) if L is not None else float('nan')
        lret_purity = compute_purity_lret(L) if L is not None else float('nan')

        rho_cirq = cirq_r.get('rho') if not cirq_r.get('oom') else None
        metrics = _compute_error_metrics(L, rho_cirq, n, CIRCUIT_DEPTH,
                                         NOISE_PROB, seed)

        cirq_status = 'ok' if not cirq_r.get('oom') else f"oom:{cirq_r.get('reason','')}"
        rank = lret['final_rank']
        row = {
            'n_qubits':       n,
            'lret_backend':   lret.get('backend', 'cpp'),
            'lret_mean_ms':   lret_ms,
            'lret_std_ms':    lret['std_ms'],
            'cirq_mean_ms':   cirq_ms,
            'cirq_std_ms':    cirq_r.get('std_ms', 0.0),
            'cirq_status':    cirq_status,
            'speedup':        speedup,
            'fidelity':       metrics['fidelity'],
            'distortion':     metrics['distortion'],
            'trace_distance': metrics['trace_distance'],
            'prob_tvd':       metrics['prob_tvd'],
            'lret_trace':     lret_trace,
            'lret_purity':    lret_purity,
            'final_rank':     rank,
            'rank_pct':       100.0 * rank / (2 ** n),
            'hilbert_dim':    2 ** n,
        }
        rows.append(row)
        _write_csv(csv_path, rows)  # crash-safe intermediate save

        print(f"LRET={lret_ms:8.1f}ms  Cirq={cirq_ms:>10.1f}ms  "
              f"speedup={speedup:6.2f}x  rank={rank:5.1f}({100*rank/(2**n):4.1f}%)  "
              f"fid={metrics['fidelity']:.4f}  distort={metrics['distortion']:.4f}  "
              f"tvd={metrics['prob_tvd']:.4f}  [Cirq: {cirq_status}]")

        # Free Cirq DM before moving on
        del cirq_r, rho_cirq, L

    print(f"\n  Final CSV: {csv_path}  ({len(rows)} rows)")

    if MATPLOTLIB_AVAILABLE and rows:
        try:
            _plot(rows, fig_base)
            print(f"  Figures: {fig_base}.{{png,pdf}}")
        except Exception as exc:
            print(f"  [warn] plotting failed: {exc}")

    return rows


# ──────────────────────────────────────────────────────────────
# Plotting (4-panel)
# ──────────────────────────────────────────────────────────────

def _plot(rows, fig_base):
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['double_col'])

    rows = [r for r in rows if 'lret_mean_ms' in r and r.get('lret_mean_ms') is not None]
    if not rows:
        return
    ns = [r['n_qubits'] for r in rows]
    lret_mean = [r['lret_mean_ms'] for r in rows]
    lret_std  = [r['lret_std_ms'] for r in rows]
    cirq_mean = [r['cirq_mean_ms'] for r in rows]
    speedup   = [r.get('speedup', float('nan')) for r in rows]
    fidelity  = [r.get('fidelity', float('nan')) for r in rows]
    rank      = [r['final_rank'] for r in rows]
    rank_pct  = [r['rank_pct'] for r in rows]

    # (a) Time
    ax = axes[0, 0]
    ax.semilogy(ns, lret_mean, 'o-', color=COLORS['lret'], label='LRET (C++)')
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

    # (b) Speedup
    ax = axes[0, 1]
    valid_s = [(n, s) for n, s in zip(ns, speedup) if np.isfinite(s)]
    if valid_s:
        ns_v, sp_v = zip(*valid_s)
        ax.plot(ns_v, sp_v, 'o-', color=COLORS['cirq_fdm'])
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1.0, label='LRET = Cirq')
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Speedup vs Cirq (x)')
    ax.legend(framealpha=0.9)
    ax.set_title('(b) Speedup Ratio')

    # (c) Fidelity (only where Cirq ran)
    ax = axes[1, 0]
    valid_f = [(n, f) for n, f in zip(ns, fidelity) if np.isfinite(f)]
    if valid_f:
        ax.plot(*zip(*valid_f), 'o-', color=COLORS['lret'])
        ax.set_ylim(min(0.99, min(f for _, f in valid_f) - 0.005), 1.001)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1.0)
    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Quantum fidelity $F(\\rho_{LRET}, \\rho_{Cirq})$')
    ax.set_title(f'(c) Approximation Fidelity ($p={NOISE_PROB}$, $\\epsilon={EPSILON}$)')

    # (d) Rank compression
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
    ax.legend(lines1 + lines2, labs1 + labs2, framealpha=0.9, loc='upper left')

    fig.suptitle('LRET vs Cirq: Same Circuit, Same Noise (Round 2)',
                 fontsize=11, fontweight='bold')
    save_figure(fig, fig_base)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='LRET vs Cirq benchmark (Round 2)')
    parser.add_argument('--quick', action='store_true', help='Reduced qubit range / trials')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()
    run(args.output_dir, args.quick)


if __name__ == '__main__':
    main()
