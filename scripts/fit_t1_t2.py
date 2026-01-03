#!/usr/bin/env python3
"""Fit T1 / T2 from decay or Ramsey measurements.

Input CSV columns (defaults):
  - t_ns: time in nanoseconds
  - p1: excited-state population (for T1)
  - coh: coherence magnitude (for T2, e.g., Ramsey fringe envelope)

Usage:
  python3 scripts/fit_t1_t2.py decay.csv --tcol t_ns --p1col p1 --output t1t2.json

Output JSON (Qiskit-style snippet for thermal_relaxation):
  {
    "errors": [
      {
        "type": "thermal_relaxation",
        "operations": ["id"],
        "gate_qubits": [[0]],
        "gate_time": 50e-9,
        "T1": 50e-6,
        "T2": 70e-6
      }
    ]
  }
"""

import argparse
import csv
import json
import math
from typing import List, Tuple


def _fit_exponential(times: List[float], values: List[float]) -> float:
    # Fit y = exp(-t/T) via linear regression on log(y)
    if len(times) != len(values) or not times:
        raise ValueError("Mismatched or empty time/value data")
    # Filter out non-positive values to avoid log issues
    filtered: List[Tuple[float, float]] = [
        (t, v) for t, v in zip(times, values) if v > 0.0
    ]
    if len(filtered) < 2:
        raise ValueError("Not enough positive samples to fit exponential")
    xs = [t for t, _ in filtered]
    ys = [math.log(v) for _, v in filtered]
    n = len(xs)
    avg_x = sum(xs) / n
    avg_y = sum(ys) / n
    num = sum((x - avg_x) * (y - avg_y) for x, y in zip(xs, ys))
    den = sum((x - avg_x) ** 2 for x in xs)
    if den == 0:
        raise ValueError("Zero variance in time samples")
    slope = num / den  # slope = -1/T
    T = -1.0 / slope if slope != 0 else float("inf")
    return T


def load_columns(path: str, tcol: str, vcol: str) -> Tuple[List[float], List[float]]:
    tvals: List[float] = []
    vvals: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tcol not in reader.fieldnames or vcol not in reader.fieldnames:
            raise ValueError(f"Columns '{tcol}' or '{vcol}' not found; available: {reader.fieldnames}")
        for row in reader:
            try:
                tvals.append(float(row[tcol]))
                vvals.append(float(row[vcol]))
            except (ValueError, TypeError):
                continue
    if not tvals or not vvals:
        raise ValueError("No valid numeric samples parsed")
    return tvals, vvals


def save_noise_json(T1: float, T2: float, gate_time: float, output_path: str) -> None:
    data = {
        "errors": [
            {
                "type": "thermal_relaxation",
                "operations": ["id"],
                "gate_qubits": [[0]],
                "gate_time": gate_time,
                "T1": T1,
                "T2": T2,
            }
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit T1/T2 from decay data")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--tcol", default="t_ns", help="Time column name (default: t_ns)")
    parser.add_argument("--p1col", default="p1", help="Excited-state population column for T1")
    parser.add_argument("--cohcol", default="coh", help="Coherence magnitude column for T2")
    parser.add_argument("--gate-time", type=float, default=50e-9, help="Gate time in seconds for export")
    parser.add_argument("--output", help="Optional output JSON path for noise model snippet")
    args = parser.parse_args()

    # Load T1 data
    t_ns, p1 = load_columns(args.csv, args.tcol, args.p1col)
    T1 = _fit_exponential(t_ns, p1) * 1e-9  # convert ns to seconds

    # Load T2 data (may be absent)
    T2 = None
    try:
        _, coh = load_columns(args.csv, args.tcol, args.cohcol)
        T2 = _fit_exponential(t_ns, coh) * 1e-9
    except Exception:
        pass

    print(f"Fitted T1 = {T1:.6e} s")
    if T2:
        print(f"Fitted T2 = {T2:.6e} s")
    else:
        print("T2 not fitted (no coherence column or insufficient data)")

    if args.output:
        save_noise_json(T1, T2 if T2 else T1, args.gate_time, args.output)
        print(f"Saved noise model JSON to {args.output}")


if __name__ == "__main__":
    main()
