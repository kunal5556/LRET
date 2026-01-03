#!/usr/bin/env python3
"""Build a simple noise model JSON from calibration CSVs.

Inputs (optional):
  --fidelity-csv: CSV with gate fidelities (column: fidelity)
  --relaxation-csv: CSV with decay data (columns: t_ns, p1, coh)

Usage examples:
  python3 scripts/calibrate_noise_model.py --fidelity-csv meas.csv -o noise.json
  python3 scripts/calibrate_noise_model.py --relaxation-csv decay.csv -o noise.json
  python3 scripts/calibrate_noise_model.py --fidelity-csv meas.csv --relaxation-csv decay.csv -o noise.json

This produces a minimal Qiskit-style noise JSON compatible with LRET import.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fit_depolarizing import estimate_depolarizing, load_fidelities
from fit_t1_t2 import _fit_exponential, load_columns


def build_noise_model(
    depol_p: float | None,
    T1: float | None,
    T2: float | None,
    gate_time: float,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []

    if depol_p is not None:
        errors.append(
            {
                "type": "depolarizing",
                "operations": ["x", "y", "z", "h"],
                "gate_qubits": [[0]],
                "param": depol_p,
            }
        )

    if T1 is not None:
        errors.append(
            {
                "type": "thermal_relaxation",
                "operations": ["id"],
                "gate_qubits": [[0]],
                "gate_time": gate_time,
                "T1": T1,
                "T2": T2 if T2 is not None else T1,
            }
        )

    return {
        "device_name": "calibrated_device",
        "noise_model_version": "1.0",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate noise model from measurement CSVs")
    parser.add_argument("--fidelity-csv", help="CSV file with gate fidelity column 'fidelity'")
    parser.add_argument("--fidelity-col", default="fidelity", help="Fidelity column name")
    parser.add_argument("--relaxation-csv", help="CSV file with decay data columns t_ns, p1, coh")
    parser.add_argument("--tcol", default="t_ns", help="Time column for decay (ns)")
    parser.add_argument("--p1col", default="p1", help="Excited-state population column for T1")
    parser.add_argument("--cohcol", default="coh", help="Coherence magnitude column for T2")
    parser.add_argument("--gate-time", type=float, default=50e-9, help="Gate time in seconds")
    parser.add_argument("-o", "--output", required=True, help="Output noise JSON file")
    args = parser.parse_args()

    depol_p = None
    if args.fidelity_csv:
        fidelities = load_fidelities(args.fidelity_csv, args.fidelity_col)
        depol_p = estimate_depolarizing(fidelities)
        print(f"Estimated depolarizing p = {depol_p:.6f}")

    T1 = T2 = None
    if args.relaxation_csv:
        t_ns, p1 = load_columns(args.relaxation_csv, args.tcol, args.p1col)
        T1 = _fit_exponential(t_ns, p1) * 1e-9
        print(f"Fitted T1 = {T1:.6e} s")
        try:
            _, coh = load_columns(args.relaxation_csv, args.tcol, args.cohcol)
            T2 = _fit_exponential(t_ns, coh) * 1e-9
            print(f"Fitted T2 = {T2:.6e} s")
        except Exception:
            pass

    if depol_p is None and T1 is None:
        raise ValueError("Provide at least one of --fidelity-csv or --relaxation-csv")

    model = build_noise_model(depol_p, T1, T2, args.gate_time)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved calibrated noise model to {output_path}")


if __name__ == "__main__":
    main()
