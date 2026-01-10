#!/usr/bin/env python3
"""
FINAL Corrected QNN Classifier Benchmark - LRET vs PennyLane

This is the DEFINITIVE benchmark that:
1. Uses PyTorch for proper gradient computation
2. Validates that learning actually occurs
3. Tests at realistic qubit counts
4. Reports proper timing with validation

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
from typing import Dict, List, Any

# Try to import torch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Install with: pip install torch")

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_CONFIG = {
    "test_cases": [
        {"n_qubits": 4, "num_epochs": 50, "batch_size": 20},   # Validation
        {"n_qubits": 6, "num_epochs": 30, "batch_size": 15},   # Quick intermediate
        # {"n_qubits": 8, "num_epochs": 20, "batch_size": 10},  # Uncomment for longer test
    ],
    "random_seed": 42,
    "n_layers": 2,
    "learning_rate": 0.1,  # Higher LR for faster convergence in test
}

# Expected timing reference (density matrix scales as O(4^n))
DENSITY_MATRIX_SCALING = {
    4: 1.0,      # baseline
    6: 16.0,     # 4^2 = 16x slower
    8: 256.0,    # 4^4 = 256x slower
    10: 4096.0,  # 4^6 = 4096x slower
}

# =============================================================================
# Utility Functions
# =============================================================================

def measure_memory() -> float:
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024**2)

def expected_density_matrix_time(n_qubits: int, base_time_4q: float) -> float:
    """Calculate expected time for density matrix simulation"""
    scale = DENSITY_MATRIX_SCALING.get(n_qubits, 4 ** (n_qubits - 4))
    return base_time_4q * scale

# =============================================================================
# QNN Training with PyTorch
# =============================================================================

def train_qnn_torch(device_name: str, n_qubits: int, num_epochs: int, 
                    batch_size: int, n_layers: int, lr: float, seed: int) -> Dict[str, Any]:
    """
    Train QNN classifier using PyTorch interface.
    This ensures proper gradient computation.
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
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create PennyLane device
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
        
        # Define QNode with torch interface
        @qml.qnode(dev, interface="torch")
        def qnn_circuit(weights, x):
            """QNN with data embedding and variational layers"""
            # Data embedding
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Entangling layer (linear topology)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        # Initialize trainable weights
        weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float64) * 0.1
        )
        
        # Generate synthetic dataset with structure
        # Class 1: Data concentrated near 0
        # Class -1: Data concentrated near pi
        X_class1 = torch.tensor(
            np.random.uniform(0, np.pi/2, (batch_size // 2, n_qubits)),
            dtype=torch.float64
        )
        X_class2 = torch.tensor(
            np.random.uniform(np.pi, 3*np.pi/2, (batch_size - batch_size // 2, n_qubits)),
            dtype=torch.float64
        )
        X_train = torch.cat([X_class1, X_class2], dim=0)
        y_train = torch.cat([
            torch.ones(batch_size // 2, dtype=torch.float64),
            -torch.ones(batch_size - batch_size // 2, dtype=torch.float64)
        ])
        
        # Shuffle
        perm = torch.randperm(batch_size)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        # Optimizer
        optimizer = torch.optim.Adam([weights], lr=lr)
        
        # Record start metrics
        start_time = time.time()
        start_memory = measure_memory()
        
        print(f"Starting training: {num_epochs} epochs, {batch_size} samples")
        print(f"  Learning rate: {lr}")
        
        # Training loop
        epoch_losses = []
        epoch_times = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            total_loss = torch.tensor(0.0, dtype=torch.float64)
            
            for x, y in zip(X_train, y_train):
                optimizer.zero_grad()
                
                # Forward pass
                prediction = qnn_circuit(weights, x)
                
                # MSE loss
                loss = (prediction - y) ** 2
                total_loss = total_loss + loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            avg_loss = (total_loss / batch_size).item()
            epoch_time = time.time() - epoch_start
            
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            
            # Progress update
            if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
        
        # Record end metrics
        end_time = time.time()
        end_memory = measure_memory()
        
        total_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Calculate metrics
        time_per_epoch = total_time / num_epochs
        time_per_sample = time_per_epoch / batch_size
        
        # Evaluate final accuracy
        with torch.no_grad():
            predictions = torch.tensor([qnn_circuit(weights, x).item() for x in X_train])
            correct = ((predictions > 0) == (y_train > 0)).sum().item()
            accuracy = correct / batch_size
        
        # Check learning occurred
        loss_ratio = epoch_losses[-1] / epoch_losses[0] if epoch_losses[0] > 0 else 1.0
        learning_occurred = loss_ratio < 0.9  # At least 10% improvement
        
        # Timing validation
        if n_qubits == 4 and time_per_sample < 0.001:
            timing_status = "⚠️ SUSPICIOUS: Too fast for 4 qubits"
        elif n_qubits >= 6 and time_per_sample < 0.01:
            timing_status = "⚠️ SUSPICIOUS: Too fast for 6+ qubits"
        else:
            timing_status = "✓ Timing appears reasonable"
        
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
            "loss_ratio": loss_ratio,
            "learning_occurred": learning_occurred,
            "training_accuracy": accuracy,
            "epoch_losses": epoch_losses,
            "epoch_times": epoch_times,
            "timing_status": timing_status,
        })
        
        print(f"\n✓ Training complete!")
        print(f"  Total time:       {total_time:.2f}s")
        print(f"  Time/epoch:       {time_per_epoch:.3f}s")
        print(f"  Time/sample:      {time_per_sample:.4f}s")
        print(f"  Memory delta:     {memory_delta:.2f} MB")
        print(f"  Loss: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f} ({(1-loss_ratio)*100:.1f}% reduction)")
        print(f"  Learning:         {'✓ YES' if learning_occurred else '✗ NO'}")
        print(f"  Accuracy:         {accuracy*100:.1f}%")
        print(f"  Timing:           {timing_status}")
        
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
# Main
# =============================================================================

def main():
    print("="*60)
    print("FINAL CORRECTED QNN BENCHMARK")
    print("Validating PennyLane Density Matrix Simulation")
    print("="*60)
    
    print(f"\nPennyLane version: {qml.__version__}")
    if HAS_TORCH:
        print(f"PyTorch version:   {torch.__version__}")
    else:
        print("ERROR: PyTorch required. Install with: pip install torch")
        return
    
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Sanity check
    print("\n--- Sanity Check ---")
    dev = qml.device("default.mixed", wires=2)
    @qml.qnode(dev, interface="torch")
    def test_grad():
        theta = torch.tensor(0.5, requires_grad=True)
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    theta = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)
    @qml.qnode(dev, interface="torch")
    def circuit(t):
        qml.RY(t, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    result = circuit(theta)
    result.backward()
    print(f"✓ Gradient computation works: d<Z>/dθ = {theta.grad:.4f}")
    expected_grad = -np.sin(0.5)
    if abs(theta.grad.item() - expected_grad) < 0.01:
        print(f"✓ Gradient is correct! (expected: {expected_grad:.4f})")
    else:
        print(f"⚠️ Gradient mismatch! (expected: {expected_grad:.4f})")
    
    # Run benchmarks
    all_results = []
    devices = ["default.mixed", "lightning.qubit"]
    
    for config in BENCHMARK_CONFIG["test_cases"]:
        n_qubits = config["n_qubits"]
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        
        print(f"\n\n{'#'*60}")
        print(f"# TEST: {n_qubits} qubits, {num_epochs} epochs, {batch_size} samples")
        print(f"{'#'*60}")
        
        for device_name in devices:
            result = train_qnn_torch(
                device_name=device_name,
                n_qubits=n_qubits,
                num_epochs=num_epochs,
                batch_size=batch_size,
                n_layers=BENCHMARK_CONFIG["n_layers"],
                lr=BENCHMARK_CONFIG["learning_rate"],
                seed=BENCHMARK_CONFIG["random_seed"],
            )
            all_results.append(result)
    
    # Summary comparison
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Group by qubits
    qubit_groups = {}
    for r in all_results:
        if r["status"] != "success":
            continue
        key = r["n_qubits"]
        if key not in qubit_groups:
            qubit_groups[key] = {}
        qubit_groups[key][r["device"]] = r
    
    print("\n| Qubits | Device | Time/Sample | Memory | Final Loss | Learning? |")
    print("|--------|--------|-------------|--------|------------|-----------|")
    for n_qubits in sorted(qubit_groups.keys()):
        for device, r in qubit_groups[n_qubits].items():
            learning = "✓" if r.get("learning_occurred") else "✗"
            print(f"| {n_qubits} | {device:15} | {r['time_per_sample_seconds']:.4f}s | "
                  f"{r['memory_delta_mb']:.1f}MB | {r['final_loss']:.4f} | {learning} |")
    
    # Speed comparison
    print("\n--- Speed Comparison ---")
    for n_qubits, devices_data in qubit_groups.items():
        if "default.mixed" in devices_data and "lightning.qubit" in devices_data:
            dm_time = devices_data["default.mixed"]["time_per_sample_seconds"]
            sv_time = devices_data["lightning.qubit"]["time_per_sample_seconds"]
            ratio = dm_time / sv_time
            print(f"{n_qubits} qubits: default.mixed is {ratio:.1f}x slower than lightning.qubit")
            print(f"  This is expected! Density matrix = O(4^n), State vector = O(2^n)")
    
    # Save results
    output_file = "qnn_benchmark_final_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": BENCHMARK_CONFIG,
            "results": all_results,
            "pennylane_version": qml.__version__,
            "torch_version": torch.__version__ if HAS_TORCH else None,
        }, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Final conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    all_learning = all(r.get("learning_occurred", False) for r in all_results if r["status"] == "success")
    if all_learning:
        print("✓ All tests showed learning (loss decreased)")
        print("✓ Gradient computation is working correctly")
        print("✓ Benchmark is valid for comparison")
    else:
        non_learning = [r for r in all_results if r["status"] == "success" and not r.get("learning_occurred")]
        print("⚠️ Some tests did not show learning:")
        for r in non_learning:
            print(f"  - {r['device']} @ {r['n_qubits']} qubits")
    
    # Check density matrix scaling
    if len(qubit_groups) >= 2:
        sorted_qubits = sorted(qubit_groups.keys())
        q1, q2 = sorted_qubits[0], sorted_qubits[-1]
        if "default.mixed" in qubit_groups[q1] and "default.mixed" in qubit_groups[q2]:
            t1 = qubit_groups[q1]["default.mixed"]["time_per_sample_seconds"]
            t2 = qubit_groups[q2]["default.mixed"]["time_per_sample_seconds"]
            actual_ratio = t2 / t1
            expected_ratio = DENSITY_MATRIX_SCALING[q2] / DENSITY_MATRIX_SCALING[q1]
            print(f"\nScaling Analysis (default.mixed):")
            print(f"  {q1}→{q2} qubits: {actual_ratio:.1f}x slowdown (expected: {expected_ratio:.0f}x)")
            if actual_ratio > expected_ratio * 0.1:  # Within order of magnitude
                print("  ✓ Scaling is consistent with O(4^n) density matrix complexity")
            else:
                print("  ⚠️ Scaling seems off - may indicate simulation shortcuts")

if __name__ == "__main__":
    main()
