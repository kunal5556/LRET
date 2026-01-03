import argparse
import json
from typing import Dict, List


THRESH_REFOCUSING = 1.05
THRESH_IDLE = 0.95


def load_process_fidelities(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def maybe_add_refocusing(rules: List[dict], pf: Dict[str, float]):
    if "X->Z" in pf and "Z_alone" in pf:
        if pf["X->Z"] > pf["Z_alone"] * THRESH_REFOCUSING:
            rules.append({
                "trigger": {"gate_type": "x"},
                "effect": {"affected_gate": "z", "error_scale": 0.7},
                "memory_depth": 1,
            })


def maybe_add_idle(rules: List[dict], pf: Dict[str, float]):
    if "idle->X" in pf and "X_alone" in pf:
        if pf["idle->X"] < pf["X_alone"] * THRESH_IDLE:
            rules.append({
                "trigger": {"gate_type": "idle"},
                "effect": {"affected_gate": "any", "error_scale": 1.3},
                "memory_depth": 1,
            })


def main():
    parser = argparse.ArgumentParser(description="Detect simple memory effects from process fidelities")
    parser.add_argument("process_json", help="Input JSON with keys like 'X->Z', 'Z_alone', 'idle->X'")
    parser.add_argument("-o", "--output", required=True, help="Output JSON with memory_effects section")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum history depth (default: 2)")
    args = parser.parse_args()

    pf = load_process_fidelities(args.process_json)
    rules: List[dict] = []
    maybe_add_refocusing(rules, pf)
    maybe_add_idle(rules, pf)

    output = {
        "memory_effects": {
            "enabled": True,
            "max_memory_depth": args.max_depth,
            "rules": rules,
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote memory effects model to {args.output} with {len(rules)} rule(s)")


if __name__ == "__main__":
    main()
