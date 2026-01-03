#!/usr/bin/env python3
"""Generate synthetic calibration data for testing calibration pipeline.

Outputs two CSV files:
  - synthetic_fidelity.csv   (columns: fidelity)
  - synthetic_relaxation.csv (columns: t_ns, p1, coh)

Usage:
  python3 scripts/generate_calibration_data.py --outdir /tmp/cal
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple


def gen_fidelities(n: int, base_f: float, jitter: float) -> List[float]:
    rng = random.Random(42)
    vals: List[float] = []
    for _ in range(n):
        f = base_f + rng.uniform(-jitter, jitter)
        f = max(0.0, min(1.0, f))
        vals.append(f)
    return vals


def gen_decay(n: int, t_max_ns: float, T1_ns: float, T2_ns: float, noise: float) -> Tuple[List[float], List[float], List[float]]:
    rng = random.Random(7)
    times: List[float] = []
    p1: List[float] = []
    coh: List[float] = []
    for i in range(n):
        t = t_max_ns * i / max(1, n - 1)
        times.append(t)
        true_p1 = math.exp(-t / T1_ns)
        true_coh = math.exp(-t / T2_ns)
        p1.append(max(0.0, true_p1 + rng.uniform(-noise, noise)))
        coh.append(max(0.0, true_coh + rng.uniform(-noise, noise)))
    return times, p1, coh


def write_fidelity_csv(path: Path, fidelities: List[float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fidelity"])
        for val in fidelities:
            writer.writerow([f"{val:.6f}"])


def write_relaxation_csv(path: Path, times: List[float], p1: List[float], coh: List[float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ns", "p1", "coh"])
        for t, p, c in zip(times, p1, coh):
            writer.writerow([f"{t:.3f}", f"{p:.6f}", f"{c:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic calibration data")
    parser.add_argument("--outdir", default="./synthetic_cal", help="Output directory")
    parser.add_argument("--base-fidelity", type=float, default=0.985, help="Base gate fidelity")
    parser.add_argument("--fidelity-jitter", type=float, default=0.003, help="Uniform jitter added to fidelity")
    parser.add_argument("--samples", type=int, default=200, help="Number of fidelity samples")
    parser.add_argument("--decay-samples", type=int, default=60, help="Number of decay samples")
    parser.add_argument("--tmax-ns", type=float, default=80000.0, help="Max time for decay (ns)")
    parser.add_argument("--T1-ns", type=float, default=50000.0, help="Ground-truth T1 (ns)")
    parser.add_argument("--T2-ns", type=float, default=70000.0, help="Ground-truth T2 (ns)")
    parser.add_argument("--decay-noise", type=float, default=0.01, help="Additive noise for decay curves")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fidelities = gen_fidelities(args.samples, args.base_fidelity, args.fidelity_jitter)
    write_fidelity_csv(outdir / "synthetic_fidelity.csv", fidelities)

    times, p1, coh = gen_decay(args.decay_samples, args.tmax_ns, args.T1_ns, args.T2_ns, args.decay_noise)
    write_relaxation_csv(outdir / "synthetic_relaxation.csv", times, p1, coh)

    print(f"Wrote {outdir/'synthetic_fidelity.csv'} and {outdir/'synthetic_relaxation.csv'}")


if __name__ == "__main__":
    main()
