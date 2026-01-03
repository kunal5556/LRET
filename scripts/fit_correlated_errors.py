import argparse
import json
import math
import pandas as pd


def estimate_zz_rate(row: pd.Series, gate_time_ns: float) -> float:
    f_joint = row["fidelity_joint"]
    f_i = row["fidelity_i"]
    f_j = row["fidelity_j"]
    f_expected = max(1e-12, f_i * f_j)
    delta_f = max(0.0, f_expected - f_joint)
    p_zz = delta_f / (2.0 * f_expected)
    if p_zz <= 0.0:
        return 0.0
    gate_time_s = gate_time_ns * 1e-9
    return math.sqrt(p_zz) / gate_time_s


def build_entry(q0: int, q1: int, zz_rate_hz: float, gate_time_ns: float) -> dict:
    # Simple ZZ-only correlated Pauli channel
    gate_time_s = gate_time_ns * 1e-9
    p_zz = min(1.0, max(0.0, (zz_rate_hz * gate_time_s) ** 2))
    return {
        "type": "correlated_pauli",
        "operations": ["cx"],
        "gate_qubits": [[q0, q1]],
        "correlation_model": "zz_coupling",
        "parameters": {
            "zz_rate_hz": zz_rate_hz,
            "gate_time_ns": gate_time_ns,
        },
        "joint_probabilities": {
            "II": 1.0 - p_zz,
            "ZZ": p_zz,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fit correlated ZZ rates from XEB-style data")
    parser.add_argument("csv", help="Input CSV with columns: qubit_i, qubit_j, fidelity_joint, fidelity_i, fidelity_j")
    parser.add_argument("--gate-time-ns", type=float, default=300.0, help="Gate duration in nanoseconds (default: 300 ns)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"qubit_i", "qubit_j", "fidelity_joint", "fidelity_i", "fidelity_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    entries = []
    for (_, row) in df.iterrows():
        q0 = int(row["qubit_i"])
        q1 = int(row["qubit_j"])
        zz_rate = estimate_zz_rate(row, args.gate_time_ns)
        entries.append(build_entry(q0, q1, zz_rate, args.gate_time_ns))

    output = {"errors": entries}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote correlated noise model to {args.output} with {len(entries)} entries")


if __name__ == "__main__":
    main()
