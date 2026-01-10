#!/usr/bin/env python3
"""
QNN Classifier Benchmark: QLRET vs Standard PennyLane Devices
=============================================================

A small, quick-to-run benchmark comparing QLRET against standard PennyLane
backends for a Quantum Neural Network classifier task.

This test is designed to:
1. Run in 5-15 minutes on a standard laptop
2. Show clear performance comparisons
3. Produce publication-ready results

Usage:
    python qnn_comparison.py
    python qnn_comparison.py --qubits 8 --trials 3
    python qnn_comparison.py --quick  # Ultra-fast test mode
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Handle imports
try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError:
    print("ERROR: PennyLane not installed. Run: pip install pennylane")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: psutil not installed - memory tracking unavailable")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QNNConfig:
    """Configuration for QNN benchmark."""
    num_qubits: int = 6
    num_layers: int = 2
    num_train_samples: int = 20
    num_test_samples: int = 10
    num_epochs: int = 5
    learning_rate: float = 0.1
    batch_size: int = 5
    num_trials: int = 3
    seed: int = 42


QUICK_CONFIG = QNNConfig(
    num_qubits=4,
    num_layers=2,
    num_train_samples=10,
    num_test_samples=5,
    num_epochs=3,
    num_trials=1,
)

STANDARD_CONFIG = QNNConfig(
    num_qubits=6,
    num_layers=2,
    num_train_samples=20,
    num_test_samples=10,
    num_epochs=5,
    num_trials=3,
)

FULL_CONFIG = QNNConfig(
    num_qubits=8,
    num_layers=3,
    num_train_samples=50,
    num_test_samples=20,
    num_epochs=10,
    num_trials=5,
)


# =============================================================================
# Device Management
# =============================================================================

def get_available_devices() -> List[str]:
    """Detect available PennyLane devices."""
    devices = []
    
    # Always try QLRET first (registered as 'qlret', not 'qlret.mixed')
    try:
        import qlret
        dev = qml.device("qlret", wires=2)
        devices.append("qlret")
        print("✓ qlret available")
    except Exception as e:
        print(f"✗ qlret not available: {e}")
    
    # Standard PennyLane devices
    try:
        dev = qml.device("default.qubit", wires=2)
        devices.append("default.qubit")
        print("✓ default.qubit available")
    except Exception:
        pass
    
    try:
        dev = qml.device("default.mixed", wires=2)
        devices.append("default.mixed")
        print("✓ default.mixed available")
    except Exception:
        pass
    
    try:
        dev = qml.device("lightning.qubit", wires=2)
        devices.append("lightning.qubit")
        print("✓ lightning.qubit available")
    except Exception:
        pass
    
    return devices


def create_device(device_name: str, num_qubits: int) -> Optional[qml.Device]:
    """Create a PennyLane device."""
    try:
        if device_name == "qlret":
            return qml.device("qlret", wires=num_qubits, epsilon=1e-4)
        else:
            return qml.device(device_name, wires=num_qubits)
    except Exception as e:
        print(f"Failed to create {device_name}: {e}")
        return None


# =============================================================================
# QNN Circuit
# =============================================================================

def create_qnn_circuit(device: qml.Device, num_qubits: int, num_layers: int):
    """Create QNN classifier circuit.
    
    Returns a function that takes (weights, features) and returns classification score.
    """
    @qml.qnode(device)
    def circuit(weights, features):
        # Feature embedding (angle encoding)
        for i in range(num_qubits):
            qml.RY(features[i % len(features)], wires=i)
        
        # Variational layers
        for layer in range(num_layers):
            # Rotation layer
            for q in range(num_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            
            # Entangling layer
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if num_qubits > 2:
                qml.CNOT(wires=[num_qubits - 1, 0])  # Circular
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


# =============================================================================
# Data Generation
# =============================================================================

def generate_classification_data(
    num_train: int,
    num_test: int,
    num_features: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data.
    
    Creates two classes that differ in their feature distributions.
    """
    rng = np.random.default_rng(seed)
    
    # Class 0: features centered around 0.3
    # Class 1: features centered around 0.7
    
    X_train = []
    y_train = []
    
    for _ in range(num_train // 2):
        X_train.append(rng.uniform(0.1, 0.5, num_features))
        y_train.append(0)
        X_train.append(rng.uniform(0.5, 0.9, num_features))
        y_train.append(1)
    
    X_test = []
    y_test = []
    
    for _ in range(num_test // 2):
        X_test.append(rng.uniform(0.1, 0.5, num_features))
        y_test.append(0)
        X_test.append(rng.uniform(0.5, 0.9, num_features))
        y_test.append(1)
    
    return (
        np.array(X_train) * np.pi,  # Scale to [0, π]
        np.array(y_train),
        np.array(X_test) * np.pi,
        np.array(y_test),
    )


# =============================================================================
# Training
# =============================================================================

def train_qnn(
    circuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: QNNConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float], float]:
    """Train QNN classifier.
    
    Returns
    -------
    weights : np.ndarray
        Trained weights.
    loss_history : List[float]
        Loss at each epoch.
    training_time : float
        Total training time in seconds.
    """
    rng = np.random.default_rng(config.seed)
    
    # Initialize weights: [layers, qubits, 2]
    weights = pnp.array(
        rng.uniform(0, 2 * np.pi, (config.num_layers, config.num_qubits, 2)),
        requires_grad=True
    )
    
    opt = qml.GradientDescentOptimizer(config.learning_rate)
    
    def cost(weights, X, y):
        """Mean squared error cost (autodiff-friendly)."""
        predictions = pnp.array([circuit(weights, x) for x in X])
        # Map targets from {0, 1} to {-1, 1} to match Pauli Z expectation
        targets = 2 * y - 1
        # Mean squared error
        loss = pnp.mean((predictions - targets) ** 2)
        return loss
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        # Mini-batch training
        indices = rng.permutation(len(X_train))
        
        for i in range(0, len(X_train), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
        
        epoch_loss = float(cost(weights, X_train, y_train))
        loss_history.append(epoch_loss)
        
        if verbose:
            print(f"    Epoch {epoch + 1}/{config.num_epochs}: Loss = {epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    
    return weights, loss_history, training_time


def evaluate_qnn(
    circuit,
    weights: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate QNN classifier.
    
    Returns
    -------
    accuracy : float
        Classification accuracy.
    inference_time : float
        Total inference time in seconds.
    """
    start_time = time.time()
    
    predictions = np.array([circuit(weights, x) for x in X_test])
    
    inference_time = time.time() - start_time
    
    # Classify: positive -> class 1, negative -> class 0
    pred_classes = (predictions > 0).astype(int)
    accuracy = np.mean(pred_classes == y_test)
    
    return accuracy, inference_time


# =============================================================================
# Benchmarking
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    device_name: str
    num_qubits: int
    num_layers: int
    trial: int
    training_time_s: float
    inference_time_s: float
    final_loss: float
    test_accuracy: float
    peak_memory_mb: float
    success: bool
    error_message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def run_benchmark(
    device_name: str,
    config: QNNConfig,
    trial: int,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run single benchmark trial."""
    
    result = BenchmarkResult(
        device_name=device_name,
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        trial=trial,
        training_time_s=0.0,
        inference_time_s=0.0,
        final_loss=0.0,
        test_accuracy=0.0,
        peak_memory_mb=0.0,
        success=False,
    )
    
    try:
        # Force garbage collection
        gc.collect()
        
        # Track memory
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)
        
        # Create device
        device = create_device(device_name, config.num_qubits)
        if device is None:
            result.error_message = f"Failed to create device: {device_name}"
            return result
        
        # Create circuit
        circuit = create_qnn_circuit(device, config.num_qubits, config.num_layers)
        
        # Generate data
        X_train, y_train, X_test, y_test = generate_classification_data(
            config.num_train_samples,
            config.num_test_samples,
            config.num_qubits,
            seed=config.seed + trial,
        )
        
        # Train
        if verbose:
            print(f"  Training on {device_name}...")
        
        weights, loss_history, training_time = train_qnn(
            circuit, X_train, y_train, config, verbose=False
        )
        
        # Evaluate
        accuracy, inference_time = evaluate_qnn(
            circuit, weights, X_test, y_test
        )
        
        # Memory tracking
        if HAS_PSUTIL:
            peak_memory = process.memory_info().rss / (1024 * 1024)
            result.peak_memory_mb = peak_memory - start_memory
        
        # Update result
        result.training_time_s = training_time
        result.inference_time_s = inference_time
        result.final_loss = loss_history[-1] if loss_history else 0.0
        result.test_accuracy = accuracy
        result.success = True
        
        if verbose:
            print(f"    ✓ Training: {training_time:.2f}s, Accuracy: {accuracy:.1%}")
        
    except Exception as e:
        result.error_message = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"    ✗ Error: {result.error_message}")
    
    return result


def run_comparison(
    config: QNNConfig,
    device_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run full comparison across all available devices.
    
    Parameters
    ----------
    config : QNNConfig
        Benchmark configuration.
    device_names : list of str, optional
        Devices to test. If None, auto-detect.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    List of BenchmarkResult
        Results for all trials.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(" QNN CLASSIFIER BENCHMARK: QLRET vs Standard PennyLane")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Qubits: {config.num_qubits}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Training samples: {config.num_train_samples}")
        print(f"  Test samples: {config.num_test_samples}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Trials: {config.num_trials}")
    
    # Get devices
    if device_names is None:
        if verbose:
            print("\nDetecting available devices...")
        device_names = get_available_devices()
    
    if not device_names:
        print("ERROR: No devices available!")
        return []
    
    if verbose:
        print(f"\nTesting devices: {', '.join(device_names)}")
    
    # Run benchmarks
    all_results = []
    
    for device_name in device_names:
        if verbose:
            print(f"\n--- {device_name} ---")
        
        for trial in range(config.num_trials):
            if verbose and config.num_trials > 1:
                print(f"\nTrial {trial + 1}/{config.num_trials}:")
            
            result = run_benchmark(device_name, config, trial, verbose)
            all_results.append(result)
    
    return all_results


# =============================================================================
# Analysis & Reporting
# =============================================================================

def analyze_results(results: List[BenchmarkResult]) -> Dict:
    """Analyze benchmark results.
    
    Returns summary statistics and comparisons.
    """
    # Group by device
    by_device = {}
    for r in results:
        if r.device_name not in by_device:
            by_device[r.device_name] = []
        if r.success:
            by_device[r.device_name].append(r)
    
    # Calculate statistics
    summary = {}
    for device, device_results in by_device.items():
        if not device_results:
            continue
        
        train_times = [r.training_time_s for r in device_results]
        accuracies = [r.test_accuracy for r in device_results]
        memories = [r.peak_memory_mb for r in device_results if r.peak_memory_mb > 0]
        
        summary[device] = {
            "num_successful": len(device_results),
            "avg_training_time_s": np.mean(train_times),
            "std_training_time_s": np.std(train_times),
            "avg_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "avg_memory_mb": np.mean(memories) if memories else 0,
        }
    
    # Compare to QLRET baseline
    comparisons = {}
    if "qlret" in summary:
        qlret_time = summary["qlret"]["avg_training_time_s"]
        for device, stats in summary.items():
            if device != "qlret":
                speedup = stats["avg_training_time_s"] / qlret_time
                comparisons[device] = {
                    "speedup_vs_qlret": speedup,
                    "qlret_faster": speedup > 1,
                }
    
    return {
        "summary": summary,
        "comparisons": comparisons,
    }


def print_results_table(results: List[BenchmarkResult], analysis: Dict):
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    
    summary = analysis["summary"]
    comparisons = analysis.get("comparisons", {})
    
    # Header
    print(f"\n{'Device':<20} {'Time (s)':<12} {'Accuracy':<12} {'Memory (MB)':<12}")
    print("-" * 56)
    
    # Sort by device name (QLRET first)
    devices = sorted(summary.keys(), key=lambda x: (0 if x.startswith("qlret") else 1, x))
    
    for device in devices:
        stats = summary[device]
        time_str = f"{stats['avg_training_time_s']:.2f} ± {stats['std_training_time_s']:.2f}"
        acc_str = f"{stats['avg_accuracy']:.1%}"
        mem_str = f"{stats['avg_memory_mb']:.1f}" if stats['avg_memory_mb'] > 0 else "N/A"
        
        print(f"{device:<20} {time_str:<12} {acc_str:<12} {mem_str:<12}")
    
    # Comparison
    if comparisons:
        print("\n" + "-" * 56)
        print("Comparison vs QLRET:")
        for device, comp in comparisons.items():
            if comp["qlret_faster"]:
                print(f"  {device}: QLRET is {comp['speedup_vs_qlret']:.2f}x FASTER")
            else:
                print(f"  {device}: QLRET is {1/comp['speedup_vs_qlret']:.2f}x slower")


def save_results(
    results: List[BenchmarkResult],
    analysis: Dict,
    output_path: Path,
):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "pennylane_version": qml.__version__,
        "results": [asdict(r) for r in results],
        "analysis": analysis,
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QNN Classifier Benchmark: QLRET vs Standard PennyLane"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run ultra-fast test (4 qubits, 1 trial)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (8 qubits, 5 trials)",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Number of qubits to test",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials per device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/qnn_comparison.json",
        help="Output file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Select configuration
    if args.quick:
        config = QUICK_CONFIG
    elif args.full:
        config = FULL_CONFIG
    else:
        config = STANDARD_CONFIG
    
    # Override with command-line arguments
    if args.qubits:
        config.num_qubits = args.qubits
    if args.trials:
        config.num_trials = args.trials
    
    verbose = not args.quiet
    
    # Run benchmark
    start_time = time.time()
    
    results = run_comparison(config, verbose=verbose)
    
    if not results:
        print("No results collected!")
        return 1
    
    # Analyze
    analysis = analyze_results(results)
    
    # Report
    if verbose:
        print_results_table(results, analysis)
    
    # Save
    save_results(results, analysis, Path(args.output))
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTotal benchmark time: {elapsed:.1f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
