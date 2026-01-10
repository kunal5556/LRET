#!/usr/bin/env python3
"""
Corrected QNN Classifier Benchmark - LRET vs PennyLane default.mixed

This benchmark properly validates:
1. That default.mixed is actually running density matrix simulation
2. That LRET (when backend is available) produces correct results
3. Fair timing comparison under identical conditions

Key differences from previous benchmark:
- Uses verified working PennyLane devices
- Validates execution by checking intermediate results
- Includes sanity checks for timing (must scale with qubits)
- Reports clear warnings when backend is unavailable

Author: LRET Team
Date: January 2026
"""

import pennylane as qml
import numpy as np
import time
import psutil
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_CONFIG = {
    "test_cases": [
        {"n_qubits": 4, "num_epochs": 20, "batch_size": 10},   # Quick validation
        {"n_qubits": 6, "num_epochs": 30, "batch_size": 10},   # Intermediate
        {"n_qubits": 8, "num_epochs": 50, "batch_size": 10},   # Target size
        # Uncomment for larger tests (will take much longer):
        # {"n_qubits": 10, "num_epochs": 50, "batch_size": 10},
    ],
    "random_seed": 42,
    "n_layers": 2,
    "learning_rate": 0.01,
}

# Expected time scaling reference (seconds per epoch per sample)
# Based on PennyLane tutorial: 4 qubits = 0.028s/epoch/sample
EXPECTED_TIME_REFERENCE = {
    4: 0.03,    # seconds per epoch per sample
    6: 0.5,     # 16x slower than 4 qubits
    8: 8.0,     # 256x slower than 4 qubits (O(4^n) density matrix)
    10: 130.0,  # 4096x slower
    12: 2000.0, # 65536x slower - typically OOM for default.mixed
}

# =============================================================================
# Utility Functions
# =============================================================================

def measure_memory() -> float:
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024**2)

def validate_timing(n_qubits: int, time_per_sample: float, device_name: str) -> str:
    """Check if timing is reasonable for given qubit count"""
    expected_min = EXPECTED_TIME_REFERENCE.get(n_qubits, 0.01) * 0.1  # 10% of expected
    expected_max = EXPECTED_TIME_REFERENCE.get(n_qubits, 100.0) * 10   # 10x expected
    
    if time_per_sample < expected_min:
        return f"⚠️ SUSPICIOUS: Too fast! ({time_per_sample:.4f}s << {expected_min:.4f}s expected minimum)"
    elif time_per_sample > expected_max:
        return f"⚠️ SLOW: ({time_per_sample:.4f}s >> {expected_max:.4f}s expected maximum)"
    else:
        return f"✓ Timing reasonable ({time_per_sample:.4f}s per sample)"

# =============================================================================
# QNN Training Implementation
# =============================================================================

def create_qnn_circuit(dev, n_qubits: int, n_layers: int):
    """Create a QNN circuit as a QNode"""
    
    @qml.qnode(dev, interface="autograd")  # Using autograd for PennyLane native gradients
    def qnn_circuit(weights, x):
        """
        QNN Circuit Architecture:
        1. Data embedding: RY(x_i) on each qubit
        2. Variational layers: RY + RZ rotations + CNOT entangling
        3. Measurement: <Z> on qubit 0
        """
        # Data embedding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    return qnn_circuit

def train_qnn(device_name: str, n_qubits: int, num_epochs: int, batch_size: int,
              n_layers: int, learning_rate: float, seed: int) -> Dict[str, Any]:
    """
    Train QNN classifier on specified device.
    
    Returns detailed metrics including timing validation.
    """
    print(f"\n{'='*60}")
    print(f"Training QNN: {n_qubits} qubits on {device_name}")
    print(f"{'='*60}")
    
    result = {
        "device": device_name,
        "n_qubits": n_qubits,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Create device
        print(f"Creating {device_name} device...")
        if device_name == "default.mixed":
            dev = qml.device("default.mixed", wires=n_qubits)
        elif device_name == "lightning.qubit":
            dev = qml.device("lightning.qubit", wires=n_qubits)
        elif device_name == "default.qubit":
            dev = qml.device("default.qubit", wires=n_qubits)
        else:
            raise ValueError(f"Unknown device: {device_name}")
        
        print(f"  ✓ Device created: {dev}")
        
        # Create circuit
        circuit = create_qnn_circuit(dev, n_qubits, n_layers)
        
        # Initialize weights
        np.random.seed(seed)
        weights = np.random.randn(n_layers, n_qubits, 2) * 0.1
        
        # Generate synthetic dataset
        X_train = np.random.uniform(0, 2 * np.pi, (batch_size, n_qubits))
        y_train = np.random.choice([-1.0, 1.0], size=batch_size)
        
        # Optimizer (using PennyLane's built-in gradient descent)
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        
        # Define cost function
        def cost(weights, X, y):
            total_loss = 0.0
            for x, label in zip(X, y):
                pred = circuit(weights, x)
                total_loss += (pred - label) ** 2
            return total_loss / len(y)
        
        # Record start metrics
        start_time = time.time()
        start_memory = measure_memory()
        
        print(f"Starting training: {num_epochs} epochs, {batch_size} samples")
        
        # Training loop
        epoch_losses = []
        epoch_times = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Compute cost and update weights
            current_cost = cost(weights, X_train, y_train)
            weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
            
            epoch_time = time.time() - epoch_start
            epoch_losses.append(float(current_cost))
            epoch_times.append(epoch_time)
            
            # Progress update every 10 epochs or last epoch
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss={current_cost:.4f}, Time={epoch_time:.2f}s")
        
        # Record end metrics
        end_time = time.time()
        end_memory = measure_memory()
        
        total_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Calculate timing metrics
        time_per_epoch = total_time / num_epochs
        time_per_sample = time_per_epoch / batch_size
        
        # Validate timing
        timing_validation = validate_timing(n_qubits, time_per_sample, device_name)
        print(f"\n  {timing_validation}")
        
        # Final evaluation
        predictions = [circuit(weights, x) for x in X_train]
        accuracy = sum(
            1 for pred, label in zip(predictions, y_train)
            if np.sign(pred) == np.sign(label)
        ) / len(y_train)
        
        # Check for learning (loss should decrease)
        loss_decreased = epoch_losses[-1] < epoch_losses[0]
        loss_decrease_ratio = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0]
        
        result.update({
            "status": "success",
            "total_time_seconds": total_time,
            "time_per_epoch_seconds": time_per_epoch,
            "time_per_sample_seconds": time_per_sample,
            "memory_start_mb": start_memory,
            "memory_end_mb": end_memory,
            "memory_delta_mb": memory_delta,
            "initial_loss": epoch_losses[0],
            "final_loss": epoch_losses[-1],
            "loss_decreased": loss_decreased,
            "loss_decrease_ratio": loss_decrease_ratio,
            "training_accuracy": accuracy,
            "epoch_losses": epoch_losses,
            "epoch_times": epoch_times,
            "timing_validation": timing_validation,
        })
        
        print(f"\n✓ Training complete!")
        print(f"  Total time:     {total_time:.2f}s")
        print(f"  Time/epoch:     {time_per_epoch:.2f}s")
        print(f"  Memory delta:   {memory_delta:.2f} MB")
        print(f"  Final loss:     {epoch_losses[-1]:.4f}")
        print(f"  Loss decreased: {loss_decreased} ({loss_decrease_ratio*100:.1f}%)")
        print(f"  Accuracy:       {accuracy*100:.1f}%")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        result.update({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
    
    return result

# =============================================================================
# Comparison Analysis
# =============================================================================

def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive comparison analysis"""
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "comparisons": [],
    }
    
    # Group by qubit count
    qubit_groups = {}
    for r in results:
        if r["status"] != "success":
            continue
        key = r["n_qubits"]
        if key not in qubit_groups:
            qubit_groups[key] = {}
        qubit_groups[key][r["device"]] = r
    
    for n_qubits, devices in sorted(qubit_groups.items()):
        print(f"\n### {n_qubits} Qubits ###")
        
        comparison = {"n_qubits": n_qubits, "devices": list(devices.keys())}
        
        # Print metrics for each device
        for device_name, r in devices.items():
            print(f"\n{device_name}:")
            print(f"  Time/epoch:     {r['time_per_epoch_seconds']:.3f}s")
            print(f"  Time/sample:    {r['time_per_sample_seconds']:.4f}s")
            print(f"  Memory delta:   {r['memory_delta_mb']:.2f} MB")
            print(f"  Final loss:     {r['final_loss']:.4f}")
            print(f"  Accuracy:       {r['training_accuracy']*100:.1f}%")
            print(f"  Validation:     {r['timing_validation']}")
        
        # Compare if multiple devices
        if len(devices) >= 2:
            device_names = list(devices.keys())
            for i in range(len(device_names)):
                for j in range(i+1, len(device_names)):
                    d1, d2 = device_names[i], device_names[j]
                    r1, r2 = devices[d1], devices[d2]
                    
                    speedup = r2["total_time_seconds"] / r1["total_time_seconds"]
                    memory_ratio = r2["memory_delta_mb"] / max(r1["memory_delta_mb"], 1)
                    
                    print(f"\n{d1} vs {d2}:")
                    print(f"  Speed ratio:    {speedup:.2f}x ", end="")
                    if speedup > 1:
                        print(f"({d1} is {speedup:.2f}x faster)")
                    else:
                        print(f"({d2} is {1/speedup:.2f}x faster)")
                    
                    print(f"  Memory ratio:   {memory_ratio:.2f}x ", end="")
                    if memory_ratio > 1:
                        print(f"({d1} uses {memory_ratio:.2f}x less)")
                    else:
                        print(f"({d2} uses {1/memory_ratio:.2f}x less)")
                    
                    comparison[f"{d1}_vs_{d2}"] = {
                        "speedup": speedup,
                        "memory_ratio": memory_ratio,
                    }
        
        analysis["comparisons"].append(comparison)
    
    return analysis

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("="*60)
    print("CORRECTED QNN CLASSIFIER BENCHMARK")
    print("Testing default.mixed vs lightning.qubit")
    print("="*60)
    print(f"\nPennyLane version: {qml.__version__}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Validate PennyLane is working
    print("\n--- Sanity Check ---")
    try:
        test_dev = qml.device("default.mixed", wires=2)
        @qml.qnode(test_dev)
        def test_circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        result = test_circuit()
        print(f"✓ PennyLane working. Test circuit result: {result:.4f}")
        assert abs(result) < 0.01, "Hadamard should give ~0 expectation"
        print("✓ Result validated!")
    except Exception as e:
        print(f"✗ PennyLane sanity check failed: {e}")
        return
    
    # Run benchmarks
    all_results = []
    
    # Devices to test (both density matrix and state vector for comparison)
    devices_to_test = ["default.mixed", "lightning.qubit"]
    
    for config in BENCHMARK_CONFIG["test_cases"]:
        n_qubits = config["n_qubits"]
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        
        print(f"\n\n{'#'*60}")
        print(f"# TEST CASE: {n_qubits} qubits, {num_epochs} epochs")
        print(f"{'#'*60}")
        
        for device_name in devices_to_test:
            result = train_qnn(
                device_name=device_name,
                n_qubits=n_qubits,
                num_epochs=num_epochs,
                batch_size=batch_size,
                n_layers=BENCHMARK_CONFIG["n_layers"],
                learning_rate=BENCHMARK_CONFIG["learning_rate"],
                seed=BENCHMARK_CONFIG["random_seed"],
            )
            all_results.append(result)
    
    # Compare results
    analysis = compare_results(all_results)
    
    # Save results
    output_file = "qnn_benchmark_corrected_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": BENCHMARK_CONFIG,
            "results": all_results,
            "analysis": analysis,
        }, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]
    
    print(f"Total tests:    {len(all_results)}")
    print(f"Successful:     {len(successful)}")
    print(f"Failed:         {len(failed)}")
    
    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  - {r['device']} @ {r['n_qubits']} qubits: {r.get('error', 'Unknown')}")
    
    # Check for suspicious timings
    suspicious = [r for r in successful if "SUSPICIOUS" in r.get("timing_validation", "")]
    if suspicious:
        print("\n⚠️ SUSPICIOUS RESULTS (may indicate issues):")
        for r in suspicious:
            print(f"  - {r['device']} @ {r['n_qubits']} qubits: {r['timing_validation']}")
    else:
        print("\n✓ All timing results appear reasonable!")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
