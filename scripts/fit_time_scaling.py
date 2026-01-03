import argparse
import json
import math
from typing import Dict
import pandas as pd
from scipy.optimize import curve_fit


def linear_model(L, r0, alpha):
    return r0 * (1.0 + alpha * L)


def exp_model(L, r0, alpha):
    return r0 * math.e ** (alpha * L)


def select_model(lengths, rates):
    # Fit linear and exponential, choose lower residual
    popt_lin, _ = curve_fit(linear_model, lengths, rates, maxfev=10000)
    residual_lin = ((rates - linear_model(lengths, *popt_lin)) ** 2).sum()

    popt_exp, _ = curve_fit(exp_model, lengths, rates, maxfev=10000)
    residual_exp = ((rates - exp_model(lengths, *popt_exp)) ** 2).sum()

    if residual_lin <= residual_exp:
        return "linear", popt_lin[0], popt_lin[1]
    return "exponential", popt_exp[0], popt_exp[1]


def fit_gate(group: pd.DataFrame) -> Dict[str, float]:
    lengths = group["sequence_length"].to_numpy(dtype=float)
    fidelities = group["avg_fidelity"].to_numpy(dtype=float)

    # Convert fidelity to per-gate depolarizing rate approximation
    rates = (1.0 - fidelities) / lengths
    model, r0, alpha = select_model(lengths, rates)
    return {
        "model": model,
        "base_rate": float(r0),
        "alpha": float(alpha),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit time-dependent noise scaling from RB data")
    parser.add_argument("csv", help="Input CSV with columns: gate_type, sequence_length, avg_fidelity")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"gate_type", "sequence_length", "avg_fidelity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = {"time_dependent_noise": {"enabled": True, "gate_parameters": {}}}
    for gate_type, group in df.groupby("gate_type"):
        fit = fit_gate(group)
        output["time_dependent_noise"]["gate_parameters"][gate_type] = {
            "base_depol_prob": fit["base_rate"],
            "alpha": fit["alpha"],
            "scaling_model": fit["model"],
            "max_depth": int(group["sequence_length"].max()),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote time-dependent noise parameters to {args.output}")


if __name__ == "__main__":
    main()
