#!/usr/bin/env python3
"""
QNN Benchmark: QLRET vs Lightning.Qubit
========================================

Parameters:
- 8 qubits
- 100 epochs  
- 50 training samples
- 3 trials for statistical significance

Expected Runtime: 2-6 hours depending on hardware

This script runs in background mode and saves intermediate results
so you can check progress.

Usage:
    python run_qnn_benchmark_8q.py

Author: LRET Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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
    num_qubits: int = 8
    num_layers: int = 3
    num_train_samples: int = 50
    num_test_samples: int = 20
    num_epochs: int = 100
    learning_rate: float = 0.1
    batch_size: int = 10
    num_trials: int = 3
    seed: int = 42


# Benchmark configuration as specified by user
BENCHMARK_CONFIG = QNNConfig(
    num_qubits=8,
    num_layers=3,
    num_train_samples=50,
    num_test_samples=20,
    num_epochs=100,
    learning_rate=0.1,
    batch_size=10,
    num_trials=3,
    seed=42,
)


# =============================================================================
# Logging
# =============================================================================

LOG_FILE = Path("benchmark_progress.log")
RESULTS_FILE = Path("results/qnn_8q_100epochs_results.json")


def log_message(msg: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def save_intermediate_results(results: List[dict], elapsed_time: float):
    """Save intermediate results to JSON file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": "in_progress",
        "elapsed_time_seconds": elapsed_time,
        "last_update": datetime.now().isoformat(),
        "config": asdict(BENCHMARK_CONFIG),
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Device Management
# =============================================================================

def create_device(device_name: str, num_qubits: int) -> Optional[qml.Device]:
    """Create a PennyLane device."""
    try:
        if device_name == "qlret":
            import qlret
            return qml.device("qlret", wires=num_qubits, epsilon=1e-4)
        else:
            return qml.device(device_name, wires=num_qubits)
    except Exception as e:
        log_message(f"Failed to create {device_name}: {e}")
        return None


# =============================================================================
# QNN Circuit
# =============================================================================

def create_qnn_circuit(device: qml.Device, num_qubits: int, num_layers: int):
    """Create QNN classifier circuit."""
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
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    
    X_train, y_train = [], []
    for _ in range(num_train // 2):
        X_train.append(rng.uniform(0.1, 0.5, num_features))
        y_train.append(0)
        X_train.append(rng.uniform(0.5, 0.9, num_features))
        y_train.append(1)
    
    X_test, y_test = [], []
    for _ in range(num_test // 2):
        X_test.append(rng.uniform(0.1, 0.5, num_features))
        y_test.append(0)
        X_test.append(rng.uniform(0.5, 0.9, num_features))
        y_test.append(1)
    
    return (
        np.array(X_train) * np.pi,
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
    device_name: str,
    trial: int,
    start_time: float,
) -> Tuple[np.ndarray, List[float], float, List[float]]:
    """Train QNN classifier with progress logging."""
    rng = np.random.default_rng(config.seed + trial)
    
    # Initialize weights: [layers, qubits, 2]
    weights = pnp.array(
        rng.uniform(0, 2 * np.pi, (config.num_layers, config.num_qubits, 2)),
        requires_grad=True
    )
    
    opt = qml.GradientDescentOptimizer(config.learning_rate)
    
    def cost(weights, X, y):
        predictions = pnp.array([circuit(weights, x) for x in X])
        targets = 2 * y - 1  # Map {0, 1} to {-1, 1}
        return pnp.mean((predictions - targets) ** 2)
    
    loss_history = []
    epoch_times = []
    train_start = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Mini-batch training
        indices = rng.permutation(len(X_train))
        
        for i in range(0, len(X_train), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
        
        epoch_loss = float(cost(weights, X_train, y_train))
        epoch_time = time.time() - epoch_start
        
        loss_history.append(epoch_loss)
        epoch_times.append(epoch_time)
        
        # Progress logging every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            total_elapsed = time.time() - start_time
            remaining_epochs = config.num_epochs - (epoch + 1)
            avg_epoch_time = np.mean(epoch_times)
            eta = remaining_epochs * avg_epoch_time
            
            log_message(
                f"  {device_name} Trial {trial+1} | Epoch {epoch+1}/{config.num_epochs} | "
                f"Loss: {epoch_loss:.4f} | Epoch time: {epoch_time:.2f}s | "
                f"ETA: {timedelta(seconds=int(eta))}"
            )
    
    training_time = time.time() - train_start
    
    return weights, loss_history, training_time, epoch_times


def evaluate_qnn(
    circuit,
    weights: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate QNN classifier."""
    start_time = time.time()
    predictions = np.array([circuit(weights, x) for x in X_test])
    inference_time = time.time() - start_time
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
    num_epochs: int
    trial: int
    training_time_s: float
    inference_time_s: float
    initial_loss: float
    final_loss: float
    test_accuracy: float
    peak_memory_mb: float
    avg_epoch_time_s: float
    success: bool
    error_message: str = ""
    timestamp: str = ""
    loss_history: List[float] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def run_benchmark(
    device_name: str,
    config: QNNConfig,
    trial: int,
    start_time: float,
) -> BenchmarkResult:
    """Run single benchmark trial."""
    
    result = BenchmarkResult(
        device_name=device_name,
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        num_epochs=config.num_epochs,
        trial=trial,
        training_time_s=0.0,
        inference_time_s=0.0,
        initial_loss=0.0,
        final_loss=0.0,
        test_accuracy=0.0,
        peak_memory_mb=0.0,
        avg_epoch_time_s=0.0,
        success=False,
    )
    
    try:
        gc.collect()
        
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)
        
        # Create device
        device = create_device(device_name, config.num_qubits)
        if device is None:
            result.error_message = f"Failed to create device: {device_name}"
            return result
        
        log_message(f"  Created device: {device}")
        
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
        weights, loss_history, training_time, epoch_times = train_qnn(
            circuit, X_train, y_train, config, device_name, trial, start_time
        )
        
        # Evaluate
        accuracy, inference_time = evaluate_qnn(circuit, weights, X_test, y_test)
        
        # Memory tracking
        if HAS_PSUTIL:
            peak_memory = process.memory_info().rss / (1024 * 1024)
            result.peak_memory_mb = peak_memory - start_memory
        
        # Update result
        result.training_time_s = training_time
        result.inference_time_s = inference_time
        result.initial_loss = loss_history[0]
        result.final_loss = loss_history[-1]
        result.test_accuracy = accuracy
        result.avg_epoch_time_s = np.mean(epoch_times)
        result.loss_history = loss_history
        result.success = True
        
        log_message(
            f"  ✓ {device_name} Trial {trial+1} complete: "
            f"Time={training_time:.1f}s, Accuracy={accuracy:.1%}, "
            f"Loss={loss_history[-1]:.4f}"
        )
        
    except Exception as e:
        import traceback
        result.error_message = f"{type(e).__name__}: {e}"
        log_message(f"  ✗ {device_name} Trial {trial+1} failed: {result.error_message}")
        traceback.print_exc()
    
    return result


def run_full_comparison(config: QNNConfig) -> List[BenchmarkResult]:
    """Run full comparison between QLRET and lightning.qubit."""
    
    log_message("=" * 70)
    log_message("QNN BENCHMARK: QLRET vs Lightning.Qubit")
    log_message("=" * 70)
    log_message(f"Configuration:")
    log_message(f"  Qubits:           {config.num_qubits}")
    log_message(f"  Layers:           {config.num_layers}")
    log_message(f"  Training samples: {config.num_train_samples}")
    log_message(f"  Test samples:     {config.num_test_samples}")
    log_message(f"  Epochs:           {config.num_epochs}")
    log_message(f"  Batch size:       {config.batch_size}")
    log_message(f"  Trials:           {config.num_trials}")
    log_message("")
    
    # Estimate time
    estimated_time_per_epoch_qlret = 5.0  # seconds (conservative estimate)
    estimated_time_per_epoch_lightning = 0.5  # seconds
    total_epochs = config.num_epochs * config.num_trials * 2  # 2 devices
    estimated_total = (
        config.num_epochs * config.num_trials * estimated_time_per_epoch_qlret +
        config.num_epochs * config.num_trials * estimated_time_per_epoch_lightning
    )
    
    log_message(f"Estimated runtime: {timedelta(seconds=int(estimated_total))} (very rough estimate)")
    log_message("")
    
    # Devices to test
    devices = ["qlret", "lightning.qubit"]
    
    all_results = []
    start_time = time.time()
    
    for device_name in devices:
        log_message(f"\n{'='*50}")
        log_message(f"TESTING: {device_name}")
        log_message(f"{'='*50}")
        
        for trial in range(config.num_trials):
            log_message(f"\n--- Trial {trial + 1}/{config.num_trials} ---")
            
            result = run_benchmark(device_name, config, trial, start_time)
            all_results.append(asdict(result))
            
            # Save intermediate results
            elapsed = time.time() - start_time
            save_intermediate_results(all_results, elapsed)
    
    return all_results


# =============================================================================
# Analysis
# =============================================================================

def analyze_and_save_results(results: List[dict], elapsed_time: float):
    """Analyze results and save final report."""
    
    log_message("\n" + "=" * 70)
    log_message("FINAL RESULTS")
    log_message("=" * 70)
    
    # Group by device
    by_device = {}
    for r in results:
        if r["device_name"] not in by_device:
            by_device[r["device_name"]] = []
        if r["success"]:
            by_device[r["device_name"]].append(r)
    
    # Summary statistics
    summary = {}
    for device, device_results in by_device.items():
        if not device_results:
            continue
        
        train_times = [r["training_time_s"] for r in device_results]
        accuracies = [r["test_accuracy"] for r in device_results]
        final_losses = [r["final_loss"] for r in device_results]
        epoch_times = [r["avg_epoch_time_s"] for r in device_results]
        memories = [r["peak_memory_mb"] for r in device_results if r["peak_memory_mb"] > 0]
        
        summary[device] = {
            "num_successful": len(device_results),
            "avg_training_time_s": np.mean(train_times),
            "std_training_time_s": np.std(train_times),
            "avg_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "avg_final_loss": np.mean(final_losses),
            "avg_epoch_time_s": np.mean(epoch_times),
            "avg_memory_mb": np.mean(memories) if memories else 0,
        }
        
        log_message(f"\n{device}:")
        log_message(f"  Training time: {np.mean(train_times):.1f} ± {np.std(train_times):.1f} s")
        log_message(f"  Accuracy:      {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
        log_message(f"  Final loss:    {np.mean(final_losses):.4f}")
        log_message(f"  Avg epoch:     {np.mean(epoch_times):.2f} s")
        if memories:
            log_message(f"  Memory delta:  {np.mean(memories):.1f} MB")
    
    # Comparison
    if "qlret" in summary and "lightning.qubit" in summary:
        qlret_time = summary["qlret"]["avg_training_time_s"]
        lightning_time = summary["lightning.qubit"]["avg_training_time_s"]
        speedup = lightning_time / qlret_time if qlret_time > 0 else 0
        
        log_message(f"\n--- Comparison ---")
        if speedup > 1:
            log_message(f"  QLRET is {speedup:.2f}x FASTER than lightning.qubit")
        else:
            log_message(f"  lightning.qubit is {1/speedup:.2f}x faster than QLRET")
        
        qlret_mem = summary["qlret"]["avg_memory_mb"]
        lightning_mem = summary["lightning.qubit"]["avg_memory_mb"]
        if qlret_mem > 0 and lightning_mem > 0:
            mem_ratio = lightning_mem / qlret_mem
            log_message(f"  Memory ratio (lightning/qlret): {mem_ratio:.2f}x")
    
    # Save final results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_data = {
        "status": "complete",
        "elapsed_time_seconds": elapsed_time,
        "completion_time": datetime.now().isoformat(),
        "config": asdict(BENCHMARK_CONFIG),
        "results": results,
        "summary": summary,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_data, f, indent=2)
    
    log_message(f"\nResults saved to: {RESULTS_FILE}")
    log_message(f"Total elapsed time: {timedelta(seconds=int(elapsed_time))}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    
    # Clear log file
    LOG_FILE.write_text(f"Benchmark started at {datetime.now().isoformat()}\n")
    
    log_message("Starting QNN Benchmark (8 qubits, 100 epochs)")
    log_message(f"PennyLane version: {qml.__version__}")
    log_message("")
    
    start_time = time.time()
    
    try:
        results = run_full_comparison(BENCHMARK_CONFIG)
        elapsed = time.time() - start_time
        analyze_and_save_results(results, elapsed)
        
        log_message("\n" + "=" * 70)
        log_message("BENCHMARK COMPLETE")
        log_message("=" * 70)
        
    except KeyboardInterrupt:
        log_message("\nBenchmark interrupted by user")
        elapsed = time.time() - start_time
        log_message(f"Elapsed time before interruption: {timedelta(seconds=int(elapsed))}")
        
    except Exception as e:
        log_message(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
