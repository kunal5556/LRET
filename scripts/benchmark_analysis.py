#!/usr/bin/env python3
"""Statistical analysis of benchmark results for LRET quantum simulator.

This module provides comprehensive statistical analysis including:
- Exponential scaling analysis with curve fitting
- Parallel speedup computation and efficiency metrics
- Accuracy threshold validation
- Performance regression detection
- Comparison between baseline and current results

Usage:
    python scripts/benchmark_analysis.py benchmark_results.csv
    python scripts/benchmark_analysis.py --compare baseline.csv current.csv
    python scripts/benchmark_analysis.py results.csv --output analysis_report.json

Examples:
    # Analyze a single benchmark run
    python scripts/benchmark_analysis.py benchmark_output/benchmark_results.csv

    # Compare current results against baseline for regression detection
    python scripts/benchmark_analysis.py current.csv --compare baseline.csv

    # Custom output path
    python scripts/benchmark_analysis.py results.csv --output my_analysis.json

Author: LRET Development Team
Date: January 2026
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Try to import numpy and scipy for advanced analysis
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Some analyses will be limited.")

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Curve fitting will be limited.")


@dataclass
class StatisticalSummary:
    """Summary statistics for a set of measurements.
    
    Attributes:
        mean: Arithmetic mean
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        median: Median value
        q25: 25th percentile (first quartile)
        q75: 75th percentile (third quartile)
        count: Number of data points
        iqr: Interquartile range (q75 - q25)
        cv: Coefficient of variation (std/mean)
    """
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    count: int
    iqr: float = 0.0
    cv: float = 0.0
    
    def __post_init__(self):
        self.iqr = self.q75 - self.q25
        self.cv = self.std / self.mean if self.mean > 0 else 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExponentialFit:
    """Results of exponential curve fitting.
    
    For model: T(n) = a * 2^(b*n)
    
    Attributes:
        a: Multiplicative coefficient
        b: Exponent coefficient (effective scaling factor)
        r_squared: Coefficient of determination (goodness of fit)
        p_value: Statistical p-value
        doubling_ratio: Time multiplier per additional qubit (2^b)
        residual_std: Standard deviation of residuals
    """
    a: float
    b: float
    r_squared: float
    p_value: float
    doubling_ratio: float
    residual_std: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def predict(self, n: int) -> float:
        """Predict execution time for n qubits."""
        return self.a * (2 ** (self.b * n))


@dataclass
class LinearFit:
    """Results of linear curve fitting.
    
    For model: T(d) = a * d + b
    
    Attributes:
        slope: Slope (time per unit depth)
        intercept: Y-intercept (overhead time)
        r_squared: Coefficient of determination
        p_value: Statistical p-value
    """
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def predict(self, d: int) -> float:
        """Predict execution time for depth d."""
        return self.slope * d + self.intercept


@dataclass
class RegressionResult:
    """Result of regression detection between baseline and current.
    
    Attributes:
        metric: Name of the metric being compared
        baseline_value: Value from baseline
        current_value: Value from current
        change_percent: Percentage change ((current - baseline) / baseline * 100)
        regression_detected: Whether a regression was detected
        threshold_percent: Threshold used for detection
        severity: Severity level (none, minor, major, critical)
    """
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    regression_detected: bool
    threshold_percent: float
    severity: str = "none"
    
    def __post_init__(self):
        if self.regression_detected:
            if abs(self.change_percent) > 50:
                self.severity = "critical"
            elif abs(self.change_percent) > 25:
                self.severity = "major"
            else:
                self.severity = "minor"
    
    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkAnalyzer:
    """Analyzes benchmark results and detects performance regressions.
    
    This class provides comprehensive statistical analysis of benchmark
    data including scaling behavior, parallel efficiency, and accuracy
    validation.
    """
    
    # Thresholds for regression detection
    REGRESSION_THRESHOLD_SCALING = 0.10  # 10% degradation
    REGRESSION_THRESHOLD_PARALLEL = 0.10  # 10% speedup loss
    REGRESSION_THRESHOLD_ACCURACY = 0.001  # Fidelity drop
    FIDELITY_THRESHOLD = 0.999  # Minimum acceptable fidelity
    
    def __init__(self, csv_path: Path):
        """Initialize analyzer with benchmark results CSV.
        
        Args:
            csv_path: Path to benchmark results CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        self.data = self._load_data()
        self.analysis_timestamp = datetime.now().isoformat()
    
    def _load_data(self) -> List[dict]:
        """Load and validate CSV data."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        # Convert numeric fields
        for row in data:
            for key in ["n_qubits", "depth", "trial", "final_rank"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        row[key] = -1
            
            for key in ["time_ms", "memory_mb", "fidelity", "trace_distance", 
                        "noise_level", "reported_time_ms"]:
                if key in row and row[key] and row[key] != "None":
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
        
        return data
    
    def get_category_data(self, category: str) -> List[dict]:
        """Filter data by category.
        
        Args:
            category: Benchmark category name
            
        Returns:
            List of rows matching the category
        """
        return [r for r in self.data if r.get("category") == category]
    
    def get_successful_data(self, category: str) -> List[dict]:
        """Get successful (non-error) results for a category."""
        return [
            r for r in self.get_category_data(category)
            if r.get("time_ms") is not None and r["time_ms"] > 0
        ]
    
    @staticmethod
    def compute_statistics(values: List[float]) -> StatisticalSummary:
        """Compute comprehensive summary statistics.
        
        Args:
            values: List of numeric values
            
        Returns:
            StatisticalSummary with computed statistics
        """
        if not values:
            return StatisticalSummary(
                mean=0, std=0, min=0, max=0, median=0,
                q25=0, q75=0, count=0
            )
        
        if HAS_NUMPY:
            arr = np.array(values)
            return StatisticalSummary(
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                median=float(np.median(arr)),
                q25=float(np.percentile(arr, 25)),
                q75=float(np.percentile(arr, 75)),
                count=len(arr),
            )
        else:
            # Pure Python fallback
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mean = sum(values) / n
            
            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            
            # Percentile computation
            def percentile(data: List[float], p: float) -> float:
                k = (len(data) - 1) * p / 100
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return data[int(k)]
                return data[int(f)] * (c - k) + data[int(c)] * (k - f)
            
            return StatisticalSummary(
                mean=mean,
                std=std,
                min=min(values),
                max=max(values),
                median=sorted_vals[n // 2] if n % 2 == 1 else 
                       (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2,
                q25=percentile(sorted_vals, 25),
                q75=percentile(sorted_vals, 75),
                count=n,
            )
    
    # -------------------------------------------------------------------------
    # Scaling Analysis
    # -------------------------------------------------------------------------
    
    def analyze_scaling(self) -> Dict[str, Any]:
        """Analyze qubit scaling behavior.
        
        Fits exponential model T(n) = a * 2^(b*n) to execution times.
        
        Returns:
            Dictionary containing:
            - per_qubit_stats: Statistics for each qubit count
            - exponential_fit: Fitted model parameters
            - doubling_time_ratio: Time multiplier per qubit
            - scaling_quality: Assessment of scaling behavior
        """
        data = self.get_successful_data("scaling")
        if not data:
            return {"error": "No scaling data available"}
        
        # Group by n_qubits
        by_n: Dict[int, List[float]] = {}
        for row in data:
            n = row["n_qubits"]
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(row["time_ms"])
        
        # Compute stats per n
        scaling_stats = {}
        for n, times in sorted(by_n.items()):
            scaling_stats[str(n)] = self.compute_statistics(times).to_dict()
        
        result = {
            "per_qubit_stats": scaling_stats,
            "qubit_range": [min(by_n.keys()), max(by_n.keys())],
        }
        
        # Fit exponential model
        if HAS_SCIPY and len(by_n) >= 3:
            ns = np.array(sorted(by_n.keys()))
            means = np.array([scaling_stats[str(n)]["mean"] for n in ns])
            
            # Log transform: log(T) = log(a) + b*n*log(2)
            log_means = np.log(means)
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(ns, log_means)
            
            # Extract parameters
            b = slope / np.log(2)  # Effective exponent
            a = np.exp(intercept)
            
            # Compute residuals
            predicted = a * (2 ** (b * ns))
            residuals = means - predicted
            residual_std = float(np.std(residuals))
            
            fit = ExponentialFit(
                a=float(a),
                b=float(b),
                r_squared=float(r_value ** 2),
                p_value=float(p_value),
                doubling_ratio=float(2 ** b),
                residual_std=residual_std,
            )
            
            result["exponential_fit"] = fit.to_dict()
            result["doubling_time_ratio"] = fit.doubling_ratio
            
            # Assess scaling quality
            if fit.r_squared > 0.99:
                quality = "excellent"
            elif fit.r_squared > 0.95:
                quality = "good"
            elif fit.r_squared > 0.90:
                quality = "acceptable"
            else:
                quality = "poor"
            
            result["scaling_quality"] = quality
            
            # Check if scaling is near-optimal (b ≈ 1.0 means 2x per qubit)
            if 0.9 <= fit.b <= 1.1:
                result["scaling_assessment"] = "near_optimal"
            elif fit.b < 0.9:
                result["scaling_assessment"] = "sub_exponential"
            else:
                result["scaling_assessment"] = "super_exponential"
        else:
            # Simple fallback without scipy
            ns = sorted(by_n.keys())
            if len(ns) >= 2:
                time_ratio = scaling_stats[str(ns[-1])]["mean"] / scaling_stats[str(ns[0])]["mean"]
                qubit_diff = ns[-1] - ns[0]
                approx_doubling = time_ratio ** (1.0 / qubit_diff)
                result["doubling_time_ratio"] = approx_doubling
        
        return result
    
    # -------------------------------------------------------------------------
    # Parallel Analysis
    # -------------------------------------------------------------------------
    
    def analyze_parallel(self) -> Dict[str, Any]:
        """Analyze parallel mode speedup.
        
        Computes speedup ratios and parallel efficiency for each mode.
        
        Returns:
            Dictionary containing:
            - mode_stats: Statistics for each parallelization mode
            - speedups: Speedup relative to sequential baseline
            - best_mode: Mode with highest speedup
            - parallel_efficiency: Efficiency metrics
        """
        data = self.get_successful_data("parallel")
        if not data:
            return {"error": "No parallel benchmark data available"}
        
        # Group by mode
        by_mode: Dict[str, List[float]] = {}
        for row in data:
            mode = row["mode"]
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(row["time_ms"])
        
        # Compute stats per mode
        mode_stats = {}
        for mode, times in by_mode.items():
            mode_stats[mode] = self.compute_statistics(times).to_dict()
        
        # Compute speedup vs sequential
        if "sequential" not in mode_stats:
            return {
                "mode_stats": mode_stats,
                "error": "No sequential baseline found for speedup computation"
            }
        
        seq_mean = mode_stats["sequential"]["mean"]
        
        speedups = {}
        for mode, stats in mode_stats.items():
            speedup = seq_mean / stats["mean"] if stats["mean"] > 0 else 0.0
            
            # Assume 4 logical cores for hybrid efficiency
            num_threads = 4 if mode == "hybrid" else 2 if mode in ["row", "column"] else 1
            efficiency = speedup / num_threads if num_threads > 0 else 0.0
            
            speedups[mode] = {
                "time_ms": stats["mean"],
                "time_std": stats["std"],
                "speedup": float(speedup),
                "efficiency": float(efficiency),
                "assumed_threads": num_threads,
            }
        
        # Find best mode
        best_mode = max(speedups.keys(), key=lambda m: speedups[m]["speedup"])
        
        # Amdahl's law analysis (estimate serial fraction)
        if "hybrid" in speedups and speedups["hybrid"]["speedup"] > 1:
            # S = 1 / (f + (1-f)/p), solve for f given S and p
            S = speedups["hybrid"]["speedup"]
            p = speedups["hybrid"]["assumed_threads"]
            # f = (1/S - 1/p) / (1 - 1/p)
            if p > 1:
                serial_fraction = (1/S - 1/p) / (1 - 1/p)
                serial_fraction = max(0, min(1, serial_fraction))  # Clamp to [0,1]
            else:
                serial_fraction = None
        else:
            serial_fraction = None
        
        return {
            "mode_stats": mode_stats,
            "speedups": speedups,
            "best_mode": best_mode,
            "best_speedup": speedups[best_mode]["speedup"],
            "serial_fraction_estimate": serial_fraction,
        }
    
    # -------------------------------------------------------------------------
    # Accuracy Analysis
    # -------------------------------------------------------------------------
    
    def analyze_accuracy(self) -> Dict[str, Any]:
        """Analyze LRET vs FDM accuracy.
        
        Validates that fidelity meets minimum threshold.
        
        Returns:
            Dictionary containing:
            - per_qubit_stats: Fidelity statistics per qubit count
            - per_noise_stats: Fidelity statistics per noise level
            - threshold: Minimum fidelity threshold
            - all_passing: Whether all results meet threshold
            - worst_fidelity: Lowest observed fidelity
        """
        data = self.get_successful_data("accuracy")
        if not data:
            return {"error": "No accuracy benchmark data available"}
        
        # Extract fidelities
        fidelities_by_n: Dict[int, List[float]] = {}
        fidelities_by_noise: Dict[float, List[float]] = {}
        all_fidelities = []
        
        for row in data:
            fid = row.get("fidelity")
            if fid is not None and fid > 0:
                n = row["n_qubits"]
                noise = row.get("noise_level", 0.0) or 0.0
                
                if n not in fidelities_by_n:
                    fidelities_by_n[n] = []
                fidelities_by_n[n].append(fid)
                
                if noise not in fidelities_by_noise:
                    fidelities_by_noise[noise] = []
                fidelities_by_noise[noise].append(fid)
                
                all_fidelities.append(fid)
        
        if not all_fidelities:
            return {"error": "No valid fidelity data found"}
        
        # Compute stats
        per_qubit_stats = {}
        for n, fids in sorted(fidelities_by_n.items()):
            per_qubit_stats[str(n)] = self.compute_statistics(fids).to_dict()
        
        per_noise_stats = {}
        for noise, fids in sorted(fidelities_by_noise.items()):
            per_noise_stats[str(noise)] = self.compute_statistics(fids).to_dict()
        
        overall = self.compute_statistics(all_fidelities)
        
        # Check threshold
        all_passing = all(fid >= self.FIDELITY_THRESHOLD for fid in all_fidelities)
        worst_fidelity = min(all_fidelities)
        failing_count = sum(1 for fid in all_fidelities if fid < self.FIDELITY_THRESHOLD)
        
        return {
            "per_qubit_stats": per_qubit_stats,
            "per_noise_stats": per_noise_stats,
            "overall_stats": overall.to_dict(),
            "threshold": self.FIDELITY_THRESHOLD,
            "all_passing": all_passing,
            "worst_fidelity": worst_fidelity,
            "failing_count": failing_count,
            "total_count": len(all_fidelities),
        }
    
    # -------------------------------------------------------------------------
    # Depth Scaling Analysis
    # -------------------------------------------------------------------------
    
    def analyze_depth_scaling(self) -> Dict[str, Any]:
        """Analyze circuit depth scaling behavior.
        
        Fits linear model T(d) = a*d + b to execution times.
        
        Returns:
            Dictionary containing:
            - per_depth_stats: Statistics for each depth value
            - linear_fit: Fitted model parameters
            - time_per_depth: Time increase per depth unit
        """
        data = self.get_successful_data("depth_scaling")
        if not data:
            return {"error": "No depth scaling data available"}
        
        # Group by depth
        by_d: Dict[int, List[float]] = {}
        for row in data:
            d = row["depth"]
            if d not in by_d:
                by_d[d] = []
            by_d[d].append(row["time_ms"])
        
        # Compute stats per depth
        depth_stats = {}
        for d, times in sorted(by_d.items()):
            depth_stats[str(d)] = self.compute_statistics(times).to_dict()
        
        result = {
            "per_depth_stats": depth_stats,
            "depth_range": [min(by_d.keys()), max(by_d.keys())],
        }
        
        # Fit linear model
        if HAS_SCIPY and len(by_d) >= 3:
            depths = np.array(sorted(by_d.keys()))
            means = np.array([depth_stats[str(d)]["mean"] for d in depths])
            
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(depths, means)
            
            fit = LinearFit(
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_value ** 2),
                p_value=float(p_value),
            )
            
            result["linear_fit"] = fit.to_dict()
            result["time_per_depth_ms"] = fit.slope
            
            # Assess linearity
            if fit.r_squared > 0.99:
                result["linearity_quality"] = "excellent"
            elif fit.r_squared > 0.95:
                result["linearity_quality"] = "good"
            elif fit.r_squared > 0.90:
                result["linearity_quality"] = "acceptable"
            else:
                result["linearity_quality"] = "poor"
        
        return result
    
    # -------------------------------------------------------------------------
    # Memory Analysis
    # -------------------------------------------------------------------------
    
    def analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage patterns.
        
        Returns:
            Dictionary containing:
            - per_qubit_stats: Memory statistics per qubit count
            - peak_memory: Maximum observed memory
            - memory_per_basis_state: Memory efficiency metric
        """
        data = self.get_successful_data("memory")
        
        # Also check other categories for memory data
        for category in ["scaling", "parallel", "depth_scaling"]:
            data.extend(self.get_successful_data(category))
        
        # Filter to rows with memory data
        data = [r for r in data if r.get("memory_mb") is not None and r["memory_mb"] > 0]
        
        if not data:
            return {"error": "No memory profiling data available"}
        
        # Group by n_qubits
        by_n: Dict[int, List[float]] = {}
        for row in data:
            n = row["n_qubits"]
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(row["memory_mb"])
        
        # Compute stats
        per_qubit_stats = {}
        for n, mems in sorted(by_n.items()):
            stats = self.compute_statistics(mems)
            per_qubit_stats[str(n)] = stats.to_dict()
            
            # Compute memory per basis state (MB / 2^n)
            basis_states = 2 ** n
            per_qubit_stats[str(n)]["memory_per_basis_state_kb"] = (stats.mean * 1024) / basis_states
        
        all_mems = [row["memory_mb"] for row in data]
        
        return {
            "per_qubit_stats": per_qubit_stats,
            "peak_memory_mb": max(all_mems),
            "overall_stats": self.compute_statistics(all_mems).to_dict(),
        }
    
    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Runs all applicable analyses and combines into single report.
        
        Returns:
            Complete analysis report dictionary
        """
        report = {
            "metadata": {
                "source_file": str(self.csv_path),
                "analysis_timestamp": self.analysis_timestamp,
                "total_benchmarks": len(self.data),
                "successful_benchmarks": len([r for r in self.data 
                                              if r.get("time_ms") and r["time_ms"] > 0]),
            },
            "categories_analyzed": [],
        }
        
        # Run all analyses
        if self.get_category_data("scaling"):
            report["scaling"] = self.analyze_scaling()
            report["categories_analyzed"].append("scaling")
        
        if self.get_category_data("parallel"):
            report["parallel"] = self.analyze_parallel()
            report["categories_analyzed"].append("parallel")
        
        if self.get_category_data("accuracy"):
            report["accuracy"] = self.analyze_accuracy()
            report["categories_analyzed"].append("accuracy")
        
        if self.get_category_data("depth_scaling"):
            report["depth_scaling"] = self.analyze_depth_scaling()
            report["categories_analyzed"].append("depth_scaling")
        
        # Memory analysis (cross-category)
        memory_analysis = self.analyze_memory()
        if "error" not in memory_analysis:
            report["memory"] = memory_analysis
            if "memory" not in report["categories_analyzed"]:
                report["categories_analyzed"].append("memory")
        
        # Overall assessment
        report["overall_assessment"] = self._generate_assessment(report)
        
        return report
    
    def _generate_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment from analysis results."""
        assessment = {
            "status": "pass",
            "warnings": [],
            "recommendations": [],
        }
        
        # Check scaling quality
        if "scaling" in report and "scaling_quality" in report["scaling"]:
            quality = report["scaling"]["scaling_quality"]
            if quality in ["poor"]:
                assessment["warnings"].append(f"Scaling fit quality is {quality}")
            
            doubling = report["scaling"].get("doubling_time_ratio", 0)
            if doubling > 2.5:
                assessment["warnings"].append(
                    f"Time growth per qubit ({doubling:.2f}x) exceeds expected 2x"
                )
        
        # Check parallel efficiency
        if "parallel" in report and "speedups" in report["parallel"]:
            best = report["parallel"].get("best_speedup", 1.0)
            if best < 2.0:
                assessment["warnings"].append(
                    f"Best parallel speedup ({best:.2f}x) is below 2x"
                )
                assessment["recommendations"].append(
                    "Review parallelization strategy or increase workload size"
                )
        
        # Check accuracy
        if "accuracy" in report:
            if not report["accuracy"].get("all_passing", True):
                assessment["status"] = "fail"
                assessment["warnings"].append(
                    f"Accuracy validation failed: {report['accuracy'].get('failing_count', '?')} "
                    f"results below {self.FIDELITY_THRESHOLD} threshold"
                )
            
            worst = report["accuracy"].get("worst_fidelity")
            if worst and worst < self.FIDELITY_THRESHOLD:
                assessment["warnings"].append(
                    f"Worst fidelity ({worst:.6f}) is below threshold"
                )
        
        # Finalize status
        if assessment["warnings"] and assessment["status"] == "pass":
            assessment["status"] = "pass_with_warnings"
        
        return assessment
    
    def save_report(self, output_path: Path):
        """Save analysis report to JSON file.
        
        Args:
            output_path: Path for output JSON file
        """
        report = self.generate_report()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Analysis report saved to: {output_path}")
        return report


def compare_benchmarks(
    baseline_path: Path, 
    current_path: Path,
    threshold_scaling: float = 0.10,
    threshold_parallel: float = 0.10,
) -> Dict[str, Any]:
    """Compare two benchmark runs for regression detection.
    
    Args:
        baseline_path: Path to baseline results CSV
        current_path: Path to current results CSV
        threshold_scaling: Regression threshold for scaling (default 10%)
        threshold_parallel: Regression threshold for speedup (default 10%)
    
    Returns:
        Comparison report with regression detection results
    """
    baseline = BenchmarkAnalyzer(baseline_path)
    current = BenchmarkAnalyzer(current_path)
    
    comparison = {
        "baseline_file": str(baseline_path),
        "current_file": str(current_path),
        "comparison_timestamp": datetime.now().isoformat(),
        "regressions": [],
        "improvements": [],
    }
    
    # Compare scaling
    if baseline.get_category_data("scaling") and current.get_category_data("scaling"):
        base_scaling = baseline.analyze_scaling()
        curr_scaling = current.analyze_scaling()
        
        if "doubling_time_ratio" in base_scaling and "doubling_time_ratio" in curr_scaling:
            base_val = base_scaling["doubling_time_ratio"]
            curr_val = curr_scaling["doubling_time_ratio"]
            change = (curr_val - base_val) / base_val * 100
            
            result = RegressionResult(
                metric="scaling_doubling_ratio",
                baseline_value=base_val,
                current_value=curr_val,
                change_percent=change,
                regression_detected=change > threshold_scaling * 100,
                threshold_percent=threshold_scaling * 100,
            )
            
            comparison["scaling"] = result.to_dict()
            
            if result.regression_detected:
                comparison["regressions"].append(result.to_dict())
            elif change < -threshold_scaling * 100:
                comparison["improvements"].append(result.to_dict())
    
    # Compare parallel speedup
    if baseline.get_category_data("parallel") and current.get_category_data("parallel"):
        base_parallel = baseline.analyze_parallel()
        curr_parallel = current.analyze_parallel()
        
        if "speedups" in base_parallel and "speedups" in curr_parallel:
            base_best = base_parallel.get("best_speedup", 1.0)
            curr_best = curr_parallel.get("best_speedup", 1.0)
            change = (curr_best - base_best) / base_best * 100
            
            # Regression is when speedup decreases
            result = RegressionResult(
                metric="parallel_best_speedup",
                baseline_value=base_best,
                current_value=curr_best,
                change_percent=change,
                regression_detected=change < -threshold_parallel * 100,
                threshold_percent=threshold_parallel * 100,
            )
            
            comparison["parallel"] = result.to_dict()
            
            if result.regression_detected:
                comparison["regressions"].append(result.to_dict())
            elif change > threshold_parallel * 100:
                comparison["improvements"].append(result.to_dict())
    
    # Compare accuracy
    if baseline.get_category_data("accuracy") and current.get_category_data("accuracy"):
        base_accuracy = baseline.analyze_accuracy()
        curr_accuracy = current.analyze_accuracy()
        
        if "worst_fidelity" in base_accuracy and "worst_fidelity" in curr_accuracy:
            base_fid = base_accuracy["worst_fidelity"]
            curr_fid = curr_accuracy["worst_fidelity"]
            change = (curr_fid - base_fid) / base_fid * 100
            
            result = RegressionResult(
                metric="accuracy_worst_fidelity",
                baseline_value=base_fid,
                current_value=curr_fid,
                change_percent=change,
                regression_detected=curr_fid < BenchmarkAnalyzer.FIDELITY_THRESHOLD,
                threshold_percent=0.1,
            )
            
            comparison["accuracy"] = result.to_dict()
            
            if result.regression_detected:
                comparison["regressions"].append(result.to_dict())
    
    # Overall status
    comparison["has_regressions"] = len(comparison["regressions"]) > 0
    comparison["has_improvements"] = len(comparison["improvements"]) > 0
    comparison["total_regressions"] = len(comparison["regressions"])
    comparison["total_improvements"] = len(comparison["improvements"])
    
    return comparison


def main():
    """Main entry point for benchmark analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze LRET benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_analysis.py results.csv                    # Analyze single run
  python benchmark_analysis.py results.csv --output report.json
  python benchmark_analysis.py current.csv --compare baseline.csv  # Regression check
        """
    )
    
    parser.add_argument(
        "results",
        type=Path,
        help="Benchmark results CSV file to analyze",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_analysis.json"),
        help="Output JSON report path (default: benchmark_analysis.json)",
    )
    
    parser.add_argument(
        "--compare",
        type=Path,
        help="Baseline CSV for regression comparison",
    )
    
    parser.add_argument(
        "--threshold-scaling",
        type=float,
        default=0.10,
        help="Regression threshold for scaling (default: 0.10 = 10%%)",
    )
    
    parser.add_argument(
        "--threshold-parallel",
        type=float,
        default=0.10,
        help="Regression threshold for parallel speedup (default: 0.10 = 10%%)",
    )
    
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary to console",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    if args.compare:
        # Comparison mode
        if not args.compare.exists():
            print(f"Error: Baseline file not found: {args.compare}")
            sys.exit(1)
        
        print(f"Comparing {args.results} against baseline {args.compare}...")
        
        comparison = compare_benchmarks(
            baseline_path=args.compare,
            current_path=args.results,
            threshold_scaling=args.threshold_scaling,
            threshold_parallel=args.threshold_parallel,
        )
        
        # Save comparison
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"Comparison report saved to: {args.output}")
        
        # Print regression summary
        if comparison["has_regressions"]:
            print("\n⚠️  REGRESSIONS DETECTED:")
            for reg in comparison["regressions"]:
                print(f"  - {reg['metric']}: {reg['change_percent']:.1f}% change "
                      f"(severity: {reg['severity']})")
            sys.exit(1)
        elif comparison["has_improvements"]:
            print("\n✅ No regressions detected. Some improvements found:")
            for imp in comparison["improvements"]:
                print(f"  - {imp['metric']}: {imp['change_percent']:+.1f}% improvement")
        else:
            print("\n✅ No regressions detected.")
        
    else:
        # Analysis mode
        print(f"Analyzing {args.results}...")
        
        analyzer = BenchmarkAnalyzer(args.results)
        report = analyzer.save_report(args.output)
        
        if args.print_summary:
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY")
            print("=" * 60)
            
            meta = report["metadata"]
            print(f"Total benchmarks: {meta['total_benchmarks']}")
            print(f"Successful: {meta['successful_benchmarks']}")
            print(f"Categories: {', '.join(report['categories_analyzed'])}")
            
            assessment = report.get("overall_assessment", {})
            status = assessment.get("status", "unknown")
            print(f"\nOverall status: {status.upper()}")
            
            for warning in assessment.get("warnings", []):
                print(f"  ⚠️  {warning}")
            
            for rec in assessment.get("recommendations", []):
                print(f"  💡 {rec}")
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
