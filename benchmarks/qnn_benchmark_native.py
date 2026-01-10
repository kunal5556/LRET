#!/usr/bin/env python3
"""
QNN Classifier Benchmark - LRET Native C++ vs PennyLane lightning.qubit
Run time: ~2-6 hours for 8 qubits, 100 epochs, 50 samples

Parameters (user-specified):
- Qubits: 8
- Epochs: 100
- Dataset size: 50
- Devices: LRET (native C++) vs lightning.qubit
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add qlret to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's autograd-compatible numpy

# Import LRET device
try:
    from qlret import QLRETDevice
    print("[OK] LRET native module loaded successfully!")
except ImportError as e:
    print(f"[FAIL] Failed to import LRET: {e}")
    sys.exit(1)

# Configuration
N_QUBITS = 8
N_EPOCHS = 100
DATASET_SIZE = 50
N_LAYERS = 3
LEARNING_RATE = 0.01
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def generate_dataset(n_samples, n_features):
    """Generate synthetic binary classification dataset."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    
    # Generate labels based on sum of first half vs second half
    y = (X[:, :n_features//2].sum(axis=1) > X[:, n_features//2:].sum(axis=1)).astype(float) * 2 - 1
    
    return X, y


def train_qnn(device_name, device, n_qubits, n_layers, X_train, y_train, epochs, lr, log_file):
    """Train QNN with gradient descent using PennyLane's autograd."""
    
    # Define the QNN circuit for this device
    @qml.qnode(device)
    def qnn_circuit(params, x):
        # Data embedding layer (angle encoding)
        for i in range(n_qubits):
            qml.RY(x[i % len(x)] * np.pi, wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            # Rotation layer
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
            
            # Entangling layer (nearest-neighbor CNOTs)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        return qml.expval(qml.PauliZ(0))
    
    # Define cost function
    def cost(params):
        loss = 0.0
        for x, y in zip(X_train, y_train):
            pred = qnn_circuit(params, x)
            loss += (pred - y) ** 2
        return loss / len(X_train)
    
    # Initialize trainable parameters using PennyLane numpy
    np.random.seed(123)
    params = pnp.array(np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2)), requires_grad=True)
    
    # Use PennyLane's built-in optimizer
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    
    history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'time': []
    }
    
    start_time = time.time()
    epoch_start = start_time
    
    print(f"\n{'='*60}")
    print(f"Training on {device_name}")
    print(f"{'='*60}")
    print(f"{'Epoch':^8} | {'Loss':^12} | {'Accuracy':^10} | {'Time (s)':^10}")
    print(f"{'-'*60}")
    
    for epoch in range(epochs):
        # Optimization step
        params, loss = opt.step_and_cost(cost, params)
        
        # Compute accuracy
        predictions = np.array([qnn_circuit(params, x) for x in X_train])
        predictions_binary = np.sign(predictions)
        accuracy = np.mean(predictions_binary == y_train)
        
        epoch_time = time.time() - epoch_start
        epoch_start = time.time()
        
        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{epoch:^8} | {float(loss):^12.6f} | {accuracy:^10.2%} | {epoch_time:^10.2f}")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{device_name},{epoch},{float(loss)},{accuracy},{epoch_time}\n")
        
        history['epoch'].append(epoch)
        history['loss'].append(float(loss))
        history['accuracy'].append(float(accuracy))
        history['time'].append(epoch_time)
    
    total_time = time.time() - start_time
    print(f"{'-'*60}")
    print(f"Total training time: {total_time:.2f} seconds")
    
    return params, history, total_time


def run_benchmark(device_name, device_fn, n_qubits, n_layers, epochs, dataset_size, log_file):
    """Run complete benchmark for a device."""
    
    result = {
        'device': device_name,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'epochs': epochs,
        'dataset_size': dataset_size,
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'error': None
    }
    
    try:
        print(f"\n{'#'*70}")
        print(f"# Device: {device_name}")
        print(f"# Qubits: {n_qubits}, Layers: {n_layers}, Epochs: {epochs}, Samples: {dataset_size}")
        print(f"{'#'*70}")
        
        # Generate dataset
        print("\nGenerating dataset...")
        X, y = generate_dataset(dataset_size, n_qubits)
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Create device
        print(f"\nCreating {device_name} device...")
        start = time.time()
        dev = device_fn(wires=n_qubits)
        device_create_time = time.time() - start
        print(f"Device created in {device_create_time:.3f}s")
        result['device_create_time'] = device_create_time
        
        # Train
        print("\nStarting training...")
        trained_params, history, total_time = train_qnn(
            device_name, dev, n_qubits, n_layers, X, y, epochs, LEARNING_RATE, log_file
        )
        
        result['status'] = 'success'
        result['total_training_time'] = total_time
        result['final_loss'] = history['loss'][-1]
        result['final_accuracy'] = history['accuracy'][-1]
        result['avg_epoch_time'] = np.mean(history['time'])
        result['history'] = history
        
        print(f"\n[OK] {device_name} completed successfully!")
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Final Accuracy: {result['final_accuracy']:.2%}")
        print(f"  Total Time: {result['total_training_time']:.2f}s")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"\n[FAIL] {device_name} FAILED: {e}")
        print(traceback.format_exc())
    
    return result


def main():
    """Main benchmark entry point."""
    
    print("="*70)
    print("QNN CLASSIFIER BENCHMARK - LRET Native C++ vs lightning.qubit")
    print("="*70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Parameters:")
    print(f"  - Qubits: {N_QUBITS}")
    print(f"  - Epochs: {N_EPOCHS}")
    print(f"  - Dataset size: {DATASET_SIZE}")
    print(f"  - Layers: {N_LAYERS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print("="*70)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Log file for real-time progress
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f'qnn_benchmark_log_{timestamp}.csv')
    with open(log_file, 'w') as f:
        f.write("timestamp,device,epoch,loss,accuracy,epoch_time\n")
    print(f"Logging to: {log_file}")
    
    results = []
    
    # Define devices to test
    devices = [
        ("LRET (Native C++)", lambda wires: QLRETDevice(wires=wires)),
        ("lightning.qubit", lambda wires: qml.device("lightning.qubit", wires=wires)),
    ]
    
    # Run benchmarks
    for device_name, device_fn in devices:
        result = run_benchmark(
            device_name=device_name,
            device_fn=device_fn,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            epochs=N_EPOCHS,
            dataset_size=DATASET_SIZE,
            log_file=log_file
        )
        results.append(result)
        
        # Save intermediate results
        results_file = os.path.join(RESULTS_DIR, f'qnn_benchmark_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for r in results:
                r_copy = r.copy()
                if 'history' in r_copy and r_copy['history']:
                    r_copy['history'] = {k: [float(v) for v in vals] for k, vals in r_copy['history'].items()}
                json_results.append(r_copy)
            json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['status'] == 'success']
    
    if len(successful) >= 2:
        lret = next((r for r in successful if 'LRET' in r['device']), None)
        lightning = next((r for r in successful if 'lightning' in r['device']), None)
        
        if lret and lightning:
            speedup = lightning['total_training_time'] / lret['total_training_time']
            print(f"\nSpeedup: {speedup:.2f}x ({'LRET faster' if speedup > 1 else 'lightning.qubit faster'})")
            print(f"\nLRET:")
            print(f"  Time: {lret['total_training_time']:.2f}s")
            print(f"  Final Accuracy: {lret['final_accuracy']:.2%}")
            print(f"\nlightning.qubit:")
            print(f"  Time: {lightning['total_training_time']:.2f}s")
            print(f"  Final Accuracy: {lightning['final_accuracy']:.2%}")
    
    for r in results:
        print(f"\n{r['device']}:")
        print(f"  Status: {r['status']}")
        if r['status'] == 'success':
            print(f"  Training Time: {r['total_training_time']:.2f}s")
            print(f"  Final Accuracy: {r['final_accuracy']:.2%}")
        else:
            print(f"  Error: {r.get('error', 'Unknown')}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {results_file}")
    print(f"End time: {datetime.now().isoformat()}")
    print("="*70)


if __name__ == "__main__":
    main()
