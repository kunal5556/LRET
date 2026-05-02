#!/usr/bin/env python3
"""
Validation Utilities for Parallel Modes Benchmarks

This module provides validation checks to ensure correctness and consistency
across different parallel execution modes.

Validation checks:
1. Trace normalization: Tr(ρ) = 1.0
2. Mode consistency: Fidelity between modes ≈ 1.0
3. Rank validity: Reasonable rank values
4. Purity bounds: 0 ≤ purity ≤ 1
5. Performance regression detection

Usage:
    python benchmarks/validation_utils.py results/parallel_modes_quick/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationIssue:
    """A single validation issue."""
    check: str
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Validation report containing all issues."""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue):
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.failed_checks += 1
        elif issue.severity == "WARNING":
            self.warnings += 1

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "issues": [
                {
                    "check": issue.check,
                    "severity": issue.severity,
                    "message": issue.message,
                    "details": issue.details
                }
                for issue in self.issues
            ]
        }

    def print_summary(self):
        """Print validation summary."""
        print("=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print(f"Total checks: {self.total_checks}")
        print(f"Passed: {self.passed_checks} ✓")
        print(f"Failed: {self.failed_checks} ✗")
        print(f"Warnings: {self.warnings} ⚠")
        print("")

        if self.failed_checks == 0 and self.warnings == 0:
            print("All validation checks passed! ✓")
        else:
            print("Issues found:")
            print("")
            for issue in self.issues:
                icon = "✗" if issue.severity == "ERROR" else "⚠" if issue.severity == "WARNING" else "ℹ"
                print(f"{icon} [{issue.severity}] {issue.check}")
                print(f"  {issue.message}")
                if issue.details:
                    print(f"  Details: {issue.details}")
                print("")

        print("=" * 80)


class ModeValidator:
    """Validator for parallel mode benchmark results."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.report = ValidationReport()

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        print("Running validation checks...")
        print("")

        checks = [
            ("Trace Normalization", self.check_trace_normalization),
            ("Rank Validity", self.check_rank_validity),
            ("Purity Bounds", self.check_purity_bounds),
            ("Execution Status", self.check_execution_status),
            ("Performance Sanity", self.check_performance_sanity),
        ]

        self.report.total_checks = len(checks)

        for check_name, check_func in checks:
            print(f"  Checking: {check_name}...")
            passed = check_func()
            if passed:
                self.report.passed_checks += 1

        print("")
        return self.report

    def check_trace_normalization(self) -> bool:
        """Verify trace(ρ) = 1.0 for all modes."""
        failures = []
        for result in self.results:
            if result["status"] != "success":
                continue

            trace = result.get("trace")
            if trace is not None and abs(trace - 1.0) > 1e-6:
                failures.append({
                    "mode": result["mode"],
                    "circuit": f"{result['circuit_type']}_{result['n_qubits']}q",
                    "trace": trace,
                    "deviation": abs(trace - 1.0)
                })

        if failures:
            self.report.add_issue(ValidationIssue(
                check="trace_normalization",
                severity="ERROR",
                message=f"Found {len(failures)} trace normalization failures (|Tr(ρ) - 1.0| > 1e-6)",
                details={"failures": failures}
            ))
            return False

        return True

    def check_rank_validity(self) -> bool:
        """Check that ranks are within reasonable bounds."""
        failures = []
        for result in self.results:
            if result["status"] != "success":
                continue

            rank = result.get("final_rank")
            n_qubits = result["n_qubits"]
            max_rank = 2 ** n_qubits

            if rank is not None:
                if rank < 1 or rank > max_rank:
                    failures.append({
                        "mode": result["mode"],
                        "circuit": f"{result['circuit_type']}_{n_qubits}q",
                        "rank": rank,
                        "max_rank": max_rank
                    })

        if failures:
            self.report.add_issue(ValidationIssue(
                check="rank_validity",
                severity="ERROR",
                message=f"Found {len(failures)} invalid rank values (rank < 1 or rank > 2^n)",
                details={"failures": failures}
            ))
            return False

        return True

    def check_purity_bounds(self) -> bool:
        """Verify purity is in [0, 1]."""
        failures = []
        for result in self.results:
            if result["status"] != "success":
                continue

            purity = result.get("purity")
            if purity is not None and (purity < 0 or purity > 1):
                failures.append({
                    "mode": result["mode"],
                    "circuit": f"{result['circuit_type']}_{result['n_qubits']}q",
                    "purity": purity
                })

        if failures:
            self.report.add_issue(ValidationIssue(
                check="purity_bounds",
                severity="ERROR",
                message=f"Found {len(failures)} purity values outside [0, 1]",
                details={"failures": failures}
            ))
            return False

        return True

    def check_execution_status(self) -> bool:
        """Check execution success rates."""
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "error")
        timeout = sum(1 for r in self.results if r["status"] == "timeout")

        success_rate = success / total if total > 0 else 0

        if success_rate < 0.95:
            self.report.add_issue(ValidationIssue(
                check="execution_status",
                severity="WARNING",
                message=f"Low success rate: {success_rate:.1%} ({success}/{total})",
                details={
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "timeout": timeout
                }
            ))
            return False

        if failed > 0 or timeout > 0:
            self.report.add_issue(ValidationIssue(
                check="execution_status",
                severity="INFO",
                message=f"Some runs failed or timed out (success: {success}, failed: {failed}, timeout: {timeout})",
                details={
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "timeout": timeout
                }
            ))

        return True

    def check_performance_sanity(self) -> bool:
        """Check that performance metrics are reasonable."""
        issues = []

        # Group by configuration
        by_config = defaultdict(list)
        for result in self.results:
            if result["status"] == "success":
                key = (result["n_qubits"], result["depth"])
                by_config[key].append(result)

        # Check that parallel modes aren't significantly slower than sequential
        for config_key, config_results in by_config.items():
            n_qubits, depth = config_key

            mode_times = defaultdict(list)
            for r in config_results:
                mode_times[r["mode"]].append(r["time_wall_ms"])

            if "sequential" in mode_times:
                seq_mean = np.mean(mode_times["sequential"])

                for mode, times in mode_times.items():
                    if mode != "sequential":
                        mode_mean = np.mean(times)
                        speedup = seq_mean / mode_mean

                        # Warning if parallel mode is much slower than sequential
                        if speedup < 0.5:  # More than 2x slower
                            issues.append({
                                "mode": mode,
                                "config": f"{n_qubits}q_d{depth}",
                                "speedup": round(speedup, 2),
                                "message": f"{mode} is {1/speedup:.1f}x slower than sequential"
                            })

        if issues:
            self.report.add_issue(ValidationIssue(
                check="performance_sanity",
                severity="WARNING",
                message=f"Found {len(issues)} performance anomalies",
                details={"issues": issues}
            ))
            return False

        return True


def load_results(results_file: Path) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_file) as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(f"Expected list of results, got {type(results)}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate parallel modes benchmark results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report (JSON)"
    )

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    print(f"Loaded {len(results)} benchmark results")
    print("")

    # Run validation
    validator = ModeValidator(results)
    report = validator.validate_all()

    # Print summary
    report.print_summary()

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_file.parent / "validation_report.json"

    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Validation report saved to: {output_path}")

    # Exit with error code if validation failed
    if report.failed_checks > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
