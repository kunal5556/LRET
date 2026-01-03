#!/usr/bin/env python3
"""Compare fidelity CSVs and report deltas.

Input CSV format (both files):
  - fidelity column (default: fidelity)

Metrics:
  - mean fidelity
  - delta (calibrated - reference)
  - absolute delta

Usage:
  python3 scripts/compare_fidelities.py ref.csv test.csv --column fidelity
"""

import argparse
import csv
from statistics import mean
from typing import List, Tuple


def load_vals(path: str, column: str) -> List[float]:
    vals: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found in {path}; available: {reader.fieldnames}")
        for row in reader:
            try:
                vals.append(float(row[column]))
            except (ValueError, TypeError):
                continue
    if not vals:
        raise ValueError(f"No numeric data in column '{column}' from {path}")
    return vals


def summarize(ref: List[float], test: List[float]) -> Tuple[float, float, float]:
    m_ref = mean(ref)
    m_test = mean(test)
    delta = m_test - m_ref
    return m_ref, m_test, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fidelity CSVs")
    parser.add_argument("ref", help="Reference fidelity CSV (e.g., IBM noise sim)")
    parser.add_argument("test", help="Test fidelity CSV (e.g., calibrated sim)")
    parser.add_argument("--column", default="fidelity", help="Fidelity column name")
    args = parser.parse_args()

    ref_vals = load_vals(args.ref, args.column)
    test_vals = load_vals(args.test, args.column)
    m_ref, m_test, delta = summarize(ref_vals, test_vals)

    print(f"Reference mean fidelity:  {m_ref:.6f}")
    print(f"Test mean fidelity:       {m_test:.6f}")
    print(f"Delta (test - ref):       {delta:.6f}")
    print(f"Absolute delta:           {abs(delta):.6f}")


if __name__ == "__main__":
    main()
