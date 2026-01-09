#!/usr/bin/env python3
"""
Automated QNN Classifier Benchmark - LRET vs PennyLane default.mixed
Tests: 8 and 10 qubits, 100 epochs, 10 samples
Duration: 15-30 minutes (realistic ML training)
"""

import pennylane as qml
import torch
import numpy as np
import time
import psutil
import json
import traceback
from datetime import datetime

def measure_memory():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024**2)

def run_qnn_training(n_qubits, device_name, num_epochs=100, batch_size=10):
    """
    Run QNN classifier training
    
    Args:
        n_qubits: Number of qubits (8 or 10)
        device_name: "lret" or "default.mixed"
        num_epochs: Training epochs (default 100 for realistic ML)
        batch_size: Number of training samples (default 10)
    
    Returns:
        Dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing {n_qubits} qubits on {device_name}")
    print(f"{'='*60}")
    
    try:
        # Create device
        if device_name == "lret":
            dev = qml.device("qlret", wires=n_qubits, epsilon=1e-4, shots=None)
            print("✓ LRET device created")
        else:
            dev = qml.device("default.mixed", wires=n_qubits)
            print("✓ default.mixed device created")
        
        # QNN circuit with embedding and variational layers
        @qml.qnode(dev, interface="torch")
        def qnn_circuit(weights, x):
            """
            QNN Circuit:
            - Data embedding layer (RY rotations)
            - 2 variational layers (RY + RZ + CNOT entangling)
            - Measurement on qubit 0
            """
            # Embedding layer: encode classical data
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            
            # Variational layers
            n_layers = 2
            for layer in range(n_layers):
                # Rotation layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        # Initialize parameters
        torch.manual_seed(42)  # Reproducibility
        n_layers = 2
        weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, requires_grad=True)
        )
        
        # Generate synthetic dataset
        # Binary classification: map high-dimensional points to {-1, +1}
        np.random.seed(42)
        X_train = torch.tensor(
            np.random.uniform(0, 2*np.pi, (batch_size, n_qubits)), 
            dtype=torch.float32
        )
        y_train = torch.tensor(
            np.random.choice([-1.0, 1.0], size=batch_size),
            dtype=torch.float32
        )
        
        # Optimizer
        optimizer = torch.optim.Adam([weights], lr=0.01)
        
        # Record start metrics
        start_time = time.time()
        start_memory = measure_memory()
        
        print(f"Starting training: {num_epochs} epochs, {batch_size} samples")
        
        # Training loop
        epoch_losses = []
        for epoch in range(num_epochs):
            epoch_start = time.time()
            batch_loss = 0.0
            
            for x, y in zip(X_train, y_train):
                optimizer.zero_grad()
                
                # Forward pass
                prediction = qnn_circuit(weights, x)
                
                # Loss: MSE between prediction and label
                loss = (prediction - y) ** 2
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
            
            avg_loss = batch_loss / batch_size
            epoch_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start
            
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
        
        # Record end metrics
        end_time = time.time()
        end_memory = measure_memory()
        
        total_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Final evaluation
        with torch.no_grad():
            predictions = [qnn_circuit(weights, x).item() for x in X_train]
            accuracy = sum(
                1 for pred, label in zip(predictions, y_train) 
                if np.sign(pred) == np.sign(label.item())
            ) / len(y_train)
        
        result = {
            "device": device_name,
            "n_qubits": n_qubits,
            "status": "success",
            "total_time_seconds": total_time,
            "time_per_epoch_seconds": total_time / num_epochs,
            "memory_used_mb": memory_used,
            "final_loss": epoch_losses[-1],
            "training_accuracy": accuracy,
            "epoch_losses": epoch_losses,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✓ Training complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Memory used: {memory_used:.2f} MB")
        print(f"  Final accuracy: {accuracy*100:.1f}%")
        
        return result
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        traceback.print_exc()
        
        return {
            "device": device_name,
            "n_qubits": n_qubits,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }

def compare_results(lret_result, baseline_result):
    """Generate comparison analysis"""
    print(f"\n{'='*60}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*60}\n")
    
    if lret_result["status"] != "success" or baseline_result["status"] != "success":
        print("⚠️ Warning: One or both tests failed. Comparison may be incomplete.\n")
        return
    
    # Performance comparison
    lret_time = lret_result["total_time_seconds"]
    baseline_time = baseline_result["total_time_seconds"]
    speedup = baseline_time / lret_time
    
    print(f"Performance:")
    print(f"  LRET:          {lret_time:.2f}s")
    print(f"  default.mixed: {baseline_time:.2f}s")
    print(f"  Speedup:       {speedup:.2f}x ", end="")
    if speedup > 1:
        print(f"(LRET is {speedup:.2f}x FASTER ✓)")
    else:
        print(f"(baseline is {1/speedup:.2f}x faster)")
    print()
    
    # Memory comparison
    lret_mem = lret_result["memory_used_mb"]
    baseline_mem = baseline_result["memory_used_mb"]
    mem_ratio = baseline_mem / lret_mem if lret_mem > 0 else 0
    
    print(f"Memory:")
    print(f"  LRET:          {lret_mem:.2f} MB")
    print(f"  default.mixed: {baseline_mem:.2f} MB")
    print(f"  Ratio:         {mem_ratio:.2f}x ", end="")
    if mem_ratio > 1:
        print(f"(LRET uses {mem_ratio:.2f}x LESS memory ✓)")
    else:
        print(f"(baseline uses {1/mem_ratio:.2f}x less memory)")
    print()
    
    # Accuracy comparison
    lret_acc = lret_result["training_accuracy"]
    baseline_acc = baseline_result["training_accuracy"]
    
    print(f"Accuracy:")
    print(f"  LRET:          {lret_acc*100:.1f}%")
    print(f"  default.mixed: {baseline_acc*100:.1f}%")
    print(f"  Difference:    {abs(lret_acc - baseline_acc)*100:.1f}%")
    if abs(lret_acc - baseline_acc) < 0.05:
        print(f"  ✓ Accuracies match (within 5%)")
    print()
    
    # Summary
    print("Summary:")
    if speedup > 1 and mem_ratio > 1:
        print("  ✓✓ LRET is FASTER and uses LESS MEMORY")
    elif speedup > 1:
        print("  ✓ LRET is FASTER (memory comparable)")
    elif mem_ratio > 1:
        print("  ✓ LRET uses LESS MEMORY (speed comparable)")
    else:
        print("  ⚠️ baseline performed better (may be expected at small scale)")
    print()

def generate_report(all_results, output_file="QNN_BENCHMARK_REPORT.md"):
    """Generate comprehensive markdown report"""
    with open(output_file, 'w') as f:
        f.write("# PennyLane QNN Classifier Benchmark Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This benchmark compares LRET PennyLane plugin against PennyLane's ")
        f.write("default.mixed device for Quantum Neural Network (QNN) classifier training.\n\n")
        
        # Test configurations
        f.write("## Test Configurations\n\n")
        f.write("| Qubits | Epochs | Batch Size | Layers | Parameters |\n")
        f.write("|--------|--------|------------|--------|------------|\n")
        for result in all_results:
            if result["status"] == "success":
                n_qubits = result["n_qubits"]
                num_epochs = result["num_epochs"]
                batch_size = result["batch_size"]
                n_params = 2 * n_qubits * 2  # 2 layers, n_qubits qubits, 2 params per qubit
                f.write(f"| {n_qubits} | {num_epochs} | {batch_size} | 2 | {n_params} |\n")
        f.write("\n")
        
        # Results table
        f.write("## Performance Results\n\n")
        f.write("| Qubits | Device | Time (s) | Memory (MB) | Accuracy | Status |\n")
        f.write("|--------|--------|----------|-------------|----------|--------|\n")
        
        for result in all_results:
            n_qubits = result["n_qubits"]
            device = result["device"]
            status = result["status"]
            
            if status == "success":
                time_s = f"{result['total_time_seconds']:.2f}"
                memory_mb = f"{result['memory_used_mb']:.2f}"
                accuracy = f"{result['training_accuracy']*100:.1f}%"
            else:
                time_s = "N/A"
                memory_mb = "N/A"
                accuracy = "N/A"
            
            f.write(f"| {n_qubits} | {device} | {time_s} | {memory_mb} | {accuracy} | {status} |\n")
        
        f.write("\n")
        
        # Comparison analysis
        f.write("## Comparison Analysis\n\n")
        
        # Group by qubit count
        qubit_counts = sorted(set(r["n_qubits"] for r in all_results))
        for n_qubits in qubit_counts:
            results_for_qubits = [r for r in all_results if r["n_qubits"] == n_qubits]
            
            lret_result = next((r for r in results_for_qubits if r["device"] == "lret"), None)
            baseline_result = next((r for r in results_for_qubits if r["device"] == "default.mixed"), None)
            
            if not lret_result or not baseline_result:
                continue
            
            f.write(f"### {n_qubits} Qubits\n\n")
            
            if lret_result["status"] == "success" and baseline_result["status"] == "success":
                lret_time = lret_result["total_time_seconds"]
                baseline_time = baseline_result["total_time_seconds"]
                speedup = baseline_time / lret_time
                
                lret_mem = lret_result["memory_used_mb"]
                baseline_mem = baseline_result["memory_used_mb"]
                mem_ratio = baseline_mem / lret_mem if lret_mem > 0 else 0
                
                f.write(f"**Performance:**\n")
                f.write(f"- LRET: {lret_time:.2f}s\n")
                f.write(f"- default.mixed: {baseline_time:.2f}s\n")
                f.write(f"- **Speedup: {speedup:.2f}x**")
                if speedup > 1:
                    f.write(f" ✓ (LRET faster)\n\n")
                else:
                    f.write(f" (baseline faster)\n\n")
                
                f.write(f"**Memory:**\n")
                f.write(f"- LRET: {lret_mem:.2f} MB\n")
                f.write(f"- default.mixed: {baseline_mem:.2f} MB\n")
                f.write(f"- **Memory Ratio: {mem_ratio:.2f}x**")
                if mem_ratio > 1:
                    f.write(f" ✓ (LRET uses less)\n\n")
                else:
                    f.write(f" (baseline uses less)\n\n")
                
                f.write(f"**Accuracy:**\n")
                f.write(f"- LRET: {lret_result['training_accuracy']*100:.1f}%\n")
                f.write(f"- default.mixed: {baseline_result['training_accuracy']*100:.1f}%\n")
                f.write(f"- Difference: {abs(lret_result['training_accuracy'] - baseline_result['training_accuracy'])*100:.1f}%\n\n")
            else:
                if lret_result["status"] != "success":
                    f.write(f"⚠️ LRET test failed: {lret_result.get('error', 'Unknown error')}\n\n")
                if baseline_result["status"] != "success":
                    f.write(f"⚠️ default.mixed test failed: {baseline_result.get('error', 'Unknown error')}\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Determine overall result
        all_success = all(r["status"] == "success" for r in all_results)
        
        if all_success:
            lret_results = [r for r in all_results if r["device"] == "lret"]
            baseline_results = [r for r in all_results if r["device"] == "default.mixed"]
            
            avg_lret_time = sum(r["total_time_seconds"] for r in lret_results) / len(lret_results)
            avg_baseline_time = sum(r["total_time_seconds"] for r in baseline_results) / len(baseline_results)
            avg_speedup = avg_baseline_time / avg_lret_time
            
            avg_lret_mem = sum(r["memory_used_mb"] for r in lret_results) / len(lret_results)
            avg_baseline_mem = sum(r["memory_used_mb"] for r in baseline_results) / len(baseline_results)
            avg_mem_ratio = avg_baseline_mem / avg_lret_mem if avg_lret_mem > 0 else 0
            
            f.write(f"**Overall Performance:**\n")
            f.write(f"- Average Speedup: {avg_speedup:.2f}x\n")
            f.write(f"- Average Memory Reduction: {avg_mem_ratio:.2f}x\n\n")
            
            if avg_speedup > 1 and avg_mem_ratio > 1:
                f.write("✓✓ **LRET demonstrates clear advantages in both speed and memory efficiency.**\n\n")
            elif avg_speedup > 1:
                f.write("✓ **LRET shows speed advantages with comparable memory usage.**\n\n")
            elif avg_mem_ratio > 1:
                f.write("✓ **LRET shows memory efficiency advantages with comparable speed.**\n\n")
            else:
                f.write("⚠️ **At this scale (8-10 qubits), default.mixed performs comparably or better. ")
                f.write("LRET's advantages typically emerge at larger scales (12+ qubits).**\n\n")
            
            f.write("**Recommendation:**\n")
            f.write("- For 8-10 qubits: Both backends perform adequately\n")
            f.write("- For 12+ qubits: LRET expected to show significant advantages\n")
            f.write("- For production: Run larger-scale tests (14-16 qubits) to confirm scalability\n\n")
        else:
            f.write("⚠️ **Some tests failed. Review error logs for details.**\n\n")
        
        f.write("---\n\n")
        f.write("**Note**: This benchmark tests QNN classifier training at small scale (8-10 qubits). ")
        f.write("LRET's advantages are more pronounced at larger scales (12-24 qubits) where tensor ")
        f.write("network compression and rank truncation provide significant memory and speed benefits.\n")
    
    print(f"\n✓ Report saved to {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PennyLane QNN Classifier Benchmark")
    print("LRET vs default.mixed")
    print("="*60)
    
    all_results = []
    
    # Test configurations (100 epochs for realistic ML training)
    test_configs = [
        {"n_qubits": 8, "num_epochs": 100, "batch_size": 10},
        {"n_qubits": 10, "num_epochs": 100, "batch_size": 10},
    ]
    
    for config in test_configs:
        n_qubits = config["n_qubits"]
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        
        # Test LRET
        lret_result = run_qnn_training(n_qubits, "lret", num_epochs, batch_size)
        all_results.append(lret_result)
        
        # Test default.mixed
        baseline_result = run_qnn_training(n_qubits, "default.mixed", num_epochs, batch_size)
        all_results.append(baseline_result)
        
        # Compare
        compare_results(lret_result, baseline_result)
    
    # Save raw results
    with open('qnn_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Raw results saved to qnn_benchmark_results.json")
    
    # Generate report
    generate_report(all_results)
    
    print("\n" + "="*60)
    print("✓✓ BENCHMARK COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  - qnn_benchmark_results.json (raw data)")
    print("  - QNN_BENCHMARK_REPORT.md (analysis report)")
