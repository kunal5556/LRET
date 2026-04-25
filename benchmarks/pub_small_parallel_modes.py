"""
Small-scale parallel-mode benchmark: N = 4 to 10, six parallel modes.

Modes compared:
    SEQUENTIAL       baseline, no OpenMP
    ROW              gate fusion + row-level OpenMP
    COLUMN           rank-level OpenMP (parallelize over low-rank columns)
    BATCH            gate fusion within a batch, truncation only on noise
                     (NEW: run_batch_parallel rewritten to fuse + fused-apply)
    HYBRID           existing adaptive path
    LAYER_PARALLEL   NEW: builds explicit disjoint-qubit layers via
                     build_parallel_layers(); per-gate execution reuses
                     the row-parallel intra-gate primitive

Outputs
-------
    results/pub_small_parallel_r3/small_parallel_<stamp>.csv
    results/pub_small_parallel_r3/small_parallel_<stamp>_time.png
    results/pub_small_parallel_r3/small_parallel_<stamp>_scaling.png
    results/pub_small_parallel_r3/small_parallel_<stamp>_correctness.png
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
    run_lret_cpp,
    compute_fidelity,
)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
QUBIT_RANGE_FULL  = [4, 5, 6, 7, 8, 9, 10]
QUBIT_RANGE_QUICK = [4, 6, 8]
PARALLEL_MODES    = ['sequential', 'row', 'column', 'batch', 'hybrid', 'layer-parallel']
THREAD_COUNTS     = [1, 4, 8]
DEPTH             = 10
NOISE_PROB        = 0.001
EPSILON           = 1e-4
N_TRIALS          = 3
WARMUP_TRIALS     = 1
CIRCUIT_SEEDS     = [42, 43, 44]

FROBENIUS_TOL = 1e-6           # Cross-mode equivalence threshold
FIDELITY_MIN  = 1.0 - 1e-8     # Cross-mode fidelity lower bound

CSV_FIELDS = [
    'n_qubits', 'depth', 'circuit_seed', 'mode', 'num_threads',
    'trials', 'mean_ms', 'std_ms', 'min_ms',
    'final_rank', 'trace', 'purity',
    'frobenius_vs_sequential', 'fidelity_vs_sequential',
    'status',
]


def _write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _time_one(circuit, n_qubits, mode, num_threads, warmup, trials):
    """Run the circuit `warmup + trials` times. Return (L_last, times_ms, final_rank)."""
    # Warmup
    for _ in range(warmup):
        _ = run_lret_cpp(circuit, n_qubits, NOISE_PROB, epsilon=EPSILON,
                        parallel_mode=mode, num_threads=num_threads,
                        export_state=False, timeout_s=600.0)
    # Timed trials
    times = []
    last_L = None
    last_rank = 0
    for _ in range(trials):
        t0 = time.perf_counter()
        L, _ms, rank = run_lret_cpp(circuit, n_qubits, NOISE_PROB, epsilon=EPSILON,
                                    parallel_mode=mode, num_threads=num_threads,
                                    export_state=True, timeout_s=600.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        last_L = L
        last_rank = rank
    return last_L, np.array(times), last_rank


def _plot_results(rows, out_prefix):
    if not MATPLOTLIB_AVAILABLE:
        return
    # time vs n, at num_threads = 8, averaged over seeds
    ok_rows = [r for r in rows if r['status'] == 'ok']

    # Plot 1: time vs n at 8 threads (or highest available)
    fig, ax = plt.subplots(figsize=(8, 5))
    target_nt = max(THREAD_COUNTS)
    by_mode = {}
    for r in ok_rows:
        if int(r['num_threads']) != target_nt and r['mode'] != 'sequential':
            continue
        # sequential is always num_threads=1
        if r['mode'] == 'sequential' and int(r['num_threads']) != 1:
            continue
        by_mode.setdefault(r['mode'], []).append((int(r['n_qubits']), float(r['mean_ms'])))
    for mode in PARALLEL_MODES:
        entries = by_mode.get(mode, [])
        if not entries:
            continue
        entries.sort()
        ns = [e[0] for e in entries]
        ms = [e[1] for e in entries]
        # Average across seeds (same n appears multiple times)
        from collections import defaultdict
        agg = defaultdict(list)
        for n, m in zip(ns, ms):
            agg[n].append(m)
        xs = sorted(agg.keys())
        ys = [float(np.mean(agg[x])) for x in xs]
        ax.plot(xs, ys, marker='o', label=mode)
    ax.set_xlabel('n_qubits')
    ax.set_ylabel('mean elapsed time (ms)')
    ax.set_yscale('log')
    ax.set_title(f'Time vs N (threads={target_nt} except sequential)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_time.png', dpi=140)
    plt.close(fig)

    # Plot 2: speedup vs threads at N=10 (or max completed N)
    n_max = max((int(r['n_qubits']) for r in ok_rows), default=0)
    if n_max > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        # baseline: SEQUENTIAL at this N
        seq_baseline = {}
        for r in ok_rows:
            if int(r['n_qubits']) == n_max and r['mode'] == 'sequential':
                seq_baseline.setdefault(int(r['circuit_seed']), []).append(float(r['mean_ms']))
        # average across seeds
        if seq_baseline:
            baseline_ms = float(np.mean([np.mean(v) for v in seq_baseline.values()]))
            for mode in PARALLEL_MODES:
                if mode == 'sequential':
                    continue
                pts = {}
                for r in ok_rows:
                    if r['mode'] == mode and int(r['n_qubits']) == n_max:
                        pts.setdefault(int(r['num_threads']), []).append(float(r['mean_ms']))
                if not pts:
                    continue
                nts = sorted(pts.keys())
                sus = [baseline_ms / float(np.mean(pts[nt])) for nt in nts]
                ax.plot(nts, sus, marker='s', label=mode)
            ax.axhline(1.0, color='grey', linestyle=':', linewidth=1)
            ax.set_xlabel('num_threads')
            ax.set_ylabel(f'speedup vs SEQUENTIAL at N={n_max}')
            ax.set_title(f'Parallel scaling at N={n_max}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f'{out_prefix}_scaling.png', dpi=140)
            plt.close(fig)

    # Plot 3: Frobenius vs n (correctness)
    fig, ax = plt.subplots(figsize=(8, 5))
    by_mode = {}
    for r in ok_rows:
        if r['mode'] == 'sequential':
            continue
        if r.get('frobenius_vs_sequential') in ('', None):
            continue
        by_mode.setdefault(r['mode'], []).append(
            (int(r['n_qubits']), float(r['frobenius_vs_sequential']))
        )
    for mode in PARALLEL_MODES:
        entries = by_mode.get(mode, [])
        if not entries:
            continue
        from collections import defaultdict
        agg = defaultdict(list)
        for n, f in entries:
            agg[n].append(f)
        xs = sorted(agg.keys())
        ys = [float(np.max(agg[x])) for x in xs]  # worst-case Frobenius per N
        ax.plot(xs, ys, marker='d', label=mode)
    ax.axhline(FROBENIUS_TOL, color='red', linestyle='--', linewidth=1, label=f'tol={FROBENIUS_TOL:.0e}')
    ax.set_xlabel('n_qubits')
    ax.set_ylabel('max Frobenius distance vs SEQUENTIAL')
    ax.set_yscale('log')
    ax.set_title('Cross-mode correctness (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_correctness.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-dir', default='results/pub_small_parallel_r3')
    args = parser.parse_args()

    qubit_range = QUBIT_RANGE_QUICK if args.quick else QUBIT_RANGE_FULL

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.output_dir, f'small_parallel_{stamp}.csv')
    out_prefix = os.path.join(args.output_dir, f'small_parallel_{stamp}')

    print(f'[Parallel-modes small-scale] qubits={qubit_range}  depth={DEPTH}  '
          f'noise={NOISE_PROB}  trials={N_TRIALS}  threads={THREAD_COUNTS}')
    print(f'  modes: {PARALLEL_MODES}')
    print(f'  csv:   {csv_path}')

    rows = []
    n_violations = 0

    for n in qubit_range:
        for seed in CIRCUIT_SEEDS:
            rng = np.random.default_rng(seed)
            circuit = build_random_dense_circuit(n, DEPTH, rng=rng)

            # Ground truth: SEQUENTIAL, single-thread, used for cross-mode comparison.
            try:
                L_ref, _, rank_ref = _time_one(circuit, n, 'sequential', 1, WARMUP_TRIALS, 1)
                rho_ref = L_ref @ L_ref.conj().T
            except Exception as exc:
                print(f'  n={n} seed={seed} SEQUENTIAL failed: {exc}')
                continue

            for mode in PARALLEL_MODES:
                # SEQUENTIAL only makes sense with threads=1
                thread_list = [1] if mode == 'sequential' else THREAD_COUNTS
                for nt in thread_list:
                    status = 'ok'
                    try:
                        L_last, times, final_rank = _time_one(
                            circuit, n, mode, nt, WARMUP_TRIALS, N_TRIALS,
                        )
                        rho = L_last @ L_last.conj().T
                        trace = float(np.trace(rho).real)
                        purity = float(np.trace(rho @ rho).real)
                        frob = float(np.linalg.norm(rho - rho_ref))
                        try:
                            fid = float(compute_fidelity(L_last, rho_ref))
                        except Exception:
                            fid = float('nan')
                        if frob > FROBENIUS_TOL and mode != 'sequential':
                            status = f'CORRECTNESS_VIOLATION (frob={frob:.2e})'
                            n_violations += 1
                    except Exception as exc:
                        times = np.array([float('nan')])
                        final_rank = 0
                        trace = float('nan')
                        purity = float('nan')
                        frob = float('nan')
                        fid = float('nan')
                        status = f'FAILED: {type(exc).__name__}: {exc}'
                        print(f'    n={n} seed={seed} mode={mode} nt={nt}: {status}')

                    row = {
                        'n_qubits': n,
                        'depth': DEPTH,
                        'circuit_seed': seed,
                        'mode': mode,
                        'num_threads': nt,
                        'trials': N_TRIALS,
                        'mean_ms': float(np.mean(times)),
                        'std_ms': float(np.std(times)),
                        'min_ms': float(np.min(times)),
                        'final_rank': final_rank,
                        'trace': trace,
                        'purity': purity,
                        'frobenius_vs_sequential': frob,
                        'fidelity_vs_sequential': fid,
                        'status': status,
                    }
                    rows.append(row)
                    _write_csv(csv_path, rows)

                    print(f'  n={n:2d} seed={seed} mode={mode:<16s} nt={nt:2d} '
                          f'mean={np.mean(times):8.2f}ms '
                          f'rank={final_rank:3d} '
                          f'frob={frob:.2e} '
                          f'fid={fid:.6f} '
                          f'[{status}]')

    print(f'\n[Done] {len(rows)} rows written. Correctness violations: {n_violations}.')
    _plot_results(rows, out_prefix)
    print(f'  csv   : {csv_path}')
    if MATPLOTLIB_AVAILABLE:
        print(f'  plots : {out_prefix}_*.png')
    if n_violations > 0:
        print('  WARNING: correctness violations detected — inspect CSV rows marked CORRECTNESS_VIOLATION')
        sys.exit(1)


if __name__ == '__main__':
    main()
