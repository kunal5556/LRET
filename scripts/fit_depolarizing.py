#!/usr/bin/env python3
"""Estimate depolarizing probability from gate fidelities.

Input CSV columns (defaults):
  - fidelity: gate fidelity in [0, 1]

Usage:
  python3 scripts/fit_depolarizing.py measurements.csv --column fidelity \
      --output depol.json

Output JSON (Qiskit-style minimal):
  {
    "errors": [
      {
        "type": "depolarizing",
        "operations": ["x", "y", "z", "h"],
        "gate_qubits": [[0]],
        "param": 0.01
      }
    ]
  }
"""

import argparse
import csv
import json
from statistics import mean
from typing import List


def estimate_depolarizing(fidelities: List[float]) -> float:
    if not fidelities:
        raise ValueError("No fidelity measurements provided")
    avg_f = mean(fidelities)
    p = max(0.0, min(1.0, 1.0 - avg_f))
    return p


def load_fidelities(path: str, column: str) -> List[float]:
    vals = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found; available: {reader.fieldnames}")
        for row in reader:
            try:
                vals.append(float(row[column]))
            except (ValueError, TypeError):
                continue
    if not vals:
        raise ValueError("No valid numeric fidelities parsed")
    return vals


def save_noise_json(p: float, output_path: str) -> None:
    data = {
        "errors": [
            {
                "type": "depolarizing",
                "operations": ["x", "y", "z", "h"],
                "gate_qubits": [[0]],
                "param": p,
            }
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate depolarizing probability from gate fidelities")
    parser.add_argument("csv", help="Input CSV file with fidelity column")
    parser.add_argument("--column", default="fidelity", help="Fidelity column name (default: fidelity)")
    parser.add_argument("--output", help="Optional output JSON path for noise model")
    args = parser.parse_args()

    fidelities = load_fidelities(args.csv, args.column)
    p = estimate_depolarizing(fidelities)
    print(f"Estimated depolarizing probability p = {p:.6f}")

    if args.output:
        save_noise_json(p, args.output)
        print(f"Saved noise model JSON to {args.output}")


if __name__ == "__main__":
    main()
