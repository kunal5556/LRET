"""
Master orchestration script for all publication benchmarks.

Runs any combination of the four pub_*.py scripts via a unified CLI.

Usage:
  python benchmarks/run_publication_benchmarks.py --benchmarks all
  python benchmarks/run_publication_benchmarks.py --benchmarks cirq memory_wall --quick
  python benchmarks/run_publication_benchmarks.py --benchmarks pennylane --output-dir results/pub/
  python benchmarks/run_publication_benchmarks.py --skip-existing --n-trials 10

Benchmarks:
  cirq         2a: LRET vs Cirq/Qiskit statevector
  memory_wall  2b: LRET vs Full Density Matrix — Memory Wall
  pennylane    2c: 20-algorithm PennyLane comparison
  row_parallel 2d: Row-parallel 6-phase optimisation
  all          Run all four (default)
"""

import os
import sys
import json
import time
import argparse
import datetime
import traceback
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

BENCHMARK_MODULES = {
    'cirq':         'benchmarks.pub_lret_vs_cirq',
    'memory_wall':  'benchmarks.pub_memory_wall',
    'pennylane':    'benchmarks.pub_pennylane_algorithms',
    'row_parallel': 'benchmarks.pub_row_parallel_optimization',
}

BENCHMARK_LABELS = {
    'cirq':         '2a: LRET vs Cirq / Qiskit Statevector',
    'memory_wall':  '2b: LRET vs Full-Density-Matrix (Memory Wall)',
    'pennylane':    '2c: PennyLane 20-Algorithm Comparison',
    'row_parallel': '2d: Row-Parallel 6-Phase Optimisation',
}

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _import_benchmark(key: str):
    """Dynamically import a pub_*.py module by registry key."""
    module_path = BENCHMARK_MODULES[key]
    # Convert dotted path to file path so we can importlib.util.spec_from_file_location
    root = Path(__file__).resolve().parent.parent
    parts = module_path.split('.')
    fpath = root.joinpath(*parts).with_suffix('.py')
    if not fpath.exists():
        raise ImportError(f"Benchmark script not found: {fpath}")
    spec = importlib.util.spec_from_file_location(module_path, fpath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_exists(output_dir: str, prefix: str) -> Optional[Path]:
    """Return the most-recent CSV matching prefix_*.csv, or None."""
    d = Path(output_dir)
    matches = sorted(d.glob(f'{prefix}_*.csv'))
    return matches[-1] if matches else None


def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


# ──────────────────────────────────────────────────────────────
# Per-benchmark runner
# ──────────────────────────────────────────────────────────────

_CSV_PREFIXES = {
    'cirq':         'lret_vs_cirq',
    'memory_wall':  'memory_wall',
    'pennylane':    'pennylane_summary',
    'row_parallel': 'row_parallel',
}


def run_benchmark(
    key: str,
    output_dir: str,
    quick: bool,
    skip_existing: bool,
    n_trials: Optional[int],
) -> Dict[str, Any]:
    """Run a single benchmark and return a status record."""
    label  = BENCHMARK_LABELS[key]
    prefix = _CSV_PREFIXES[key]
    record: Dict[str, Any] = {
        'key':        key,
        'label':      label,
        'status':     'pending',
        'duration_s': 0.0,
        'output_csv': None,
        'error':      None,
    }

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Skip if CSV already exists
    if skip_existing:
        existing = _csv_exists(output_dir, prefix)
        if existing:
            print(f"  [skip] Found existing: {existing.name}")
            record['status']     = 'skipped'
            record['output_csv'] = str(existing)
            return record

    t0 = time.perf_counter()
    try:
        mod = _import_benchmark(key)

        # Override n_trials if requested (each module accepts it via run())
        kwargs: Dict[str, Any] = {'output_dir': output_dir, 'quick': quick}
        if n_trials is not None:
            sig = getattr(mod.run, '__code__', None)
            if sig and 'n_trials' in (sig.co_varnames or []):
                kwargs['n_trials'] = n_trials

        rows = mod.run(**kwargs)

        record['status']     = 'success'
        record['n_rows']     = len(rows) if rows else 0
        # Find the CSV that was just written
        fresh = _csv_exists(output_dir, prefix)
        record['output_csv'] = str(fresh) if fresh else None

    except Exception as exc:
        record['status'] = 'error'
        record['error']  = traceback.format_exc()
        print(f"\n  ERROR in {key}:\n{record['error']}")

    record['duration_s'] = time.perf_counter() - t0
    print(f"\n  Finished in {_hms(record['duration_s'])}  —  status: {record['status']}")
    return record


# ──────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────

def generate_report(
    records: List[Dict[str, Any]],
    output_dir: str,
    datestamp: str,
) -> str:
    """Write summary_stats.json + report.md; return report path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── JSON summary ──────────────────────────────────────────
    summary = {
        'run_date':    datestamp,
        'total_time':  _hms(sum(r['duration_s'] for r in records)),
        'benchmarks':  records,
        'n_success':   sum(1 for r in records if r['status'] == 'success'),
        'n_skipped':   sum(1 for r in records if r['status'] == 'skipped'),
        'n_error':     sum(1 for r in records if r['status'] == 'error'),
    }
    json_path = out / f'summary_stats_{datestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Markdown report ───────────────────────────────────────
    md_lines = [
        f'# Publication Benchmark Report — {datestamp}',
        '',
        f'Total runtime: {summary["total_time"]}',
        f'Benchmarks: {summary["n_success"]} OK / '
        f'{summary["n_skipped"]} skipped / {summary["n_error"]} error',
        '',
        '## Results',
        '',
    ]
    for r in records:
        icon = {'success': '✅', 'skipped': '⏭', 'error': '❌'}.get(r['status'], '?')
        md_lines.append(f'### {icon} {r["label"]}')
        md_lines.append('')
        md_lines.append(f'- **Status**: {r["status"]}')
        md_lines.append(f'- **Duration**: {_hms(r["duration_s"])}')
        if r.get('output_csv'):
            md_lines.append(f'- **CSV**: `{r["output_csv"]}`')
        if r['status'] == 'error' and r.get('error'):
            md_lines.append('')
            md_lines.append('```')
            md_lines.append(r['error'][:500])
            md_lines.append('```')
        md_lines.append('')

    # Figure references
    md_lines += [
        '## Publication Figures',
        '',
        'Figures are saved as PDF (vector) + PNG (300 dpi) in `output_dir`:',
        '',
        '| Script | Figure base name |',
        '|--------|-----------------|',
        '| pub_lret_vs_cirq.py        | `lret_vs_cirq_YYYYMMDD`        |',
        '| pub_memory_wall.py         | `memory_wall_YYYYMMDD`          |',
        '| pub_pennylane_algorithms.py| `pennylane_summary_YYYYMMDD`    |',
        '| pub_row_parallel_optim.py  | `row_parallel_YYYYMMDD`         |',
        '',
        '_Generated by run_publication_benchmarks.py_',
    ]

    md_path = out / f'report_{datestamp}.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')

    print(f"\n  Report : {md_path}")
    print(f"  Summary: {json_path}")
    return str(md_path)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run publication benchmarks for the LRET paper.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--benchmarks', nargs='+',
        choices=list(BENCHMARK_MODULES.keys()) + ['all'],
        default=['all'],
        help='Which benchmarks to run (default: all)',
    )
    parser.add_argument('--quick', action='store_true',
                        help='Reduced qubit range + fewer trials for testing')
    parser.add_argument('--output-dir', default='results',
                        help='Directory for CSVs, figures, and reports')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip benchmark if its CSV already exists in output-dir')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override number of trials per benchmark')
    args = parser.parse_args()

    # Resolve benchmark list
    keys = list(BENCHMARK_MODULES.keys()) if 'all' in args.benchmarks else args.benchmarks

    datestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'publication_{datestamp}') \
                 if args.output_dir == 'results' else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nPublication Benchmark Runner")
    print(f"  Benchmarks : {', '.join(keys)}")
    print(f"  Output dir : {output_dir}")
    print(f"  Quick mode : {args.quick}")
    print(f"  Skip exist.: {args.skip_existing}")

    records: List[Dict[str, Any]] = []
    total_t0 = time.perf_counter()

    for key in keys:
        rec = run_benchmark(
            key=key,
            output_dir=output_dir,
            quick=args.quick,
            skip_existing=args.skip_existing,
            n_trials=args.n_trials,
        )
        records.append(rec)

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  All benchmarks complete — total {_hms(total_elapsed)}")
    print(f"{'='*70}")
    for r in records:
        icon = {'success': '✅', 'skipped': '⏭', 'error': '❌'}.get(r['status'], '?')
        print(f"  {icon}  {r['label']:<45} {_hms(r['duration_s'])}")

    generate_report(records, output_dir, datestamp)


if __name__ == '__main__':
    import importlib.util  # ensure available at module level for _import_benchmark
    main()
