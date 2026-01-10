#!/usr/bin/env python3
"""
QNN Classifier Benchmark - LRET vs default.mixed (Noisy Simulation)
This is a FAIR comparison: both are density matrix simulators.

Parameters:
- Qubits: 10
- Noise: 0.1 (depolarizing noise after each gate)
- Epochs: 100
- Dataset size: 100
- Devices: LRET (native C++) vs default.mixed

Expected runtime: Several hours (default.mixed is slow at 10 qubits)
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
from pennylane import numpy as pnp

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
NOISE_RATE = 0.1  # 10% depolarizing noise
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def generate_dataset(n_samples, n_features):
    """Generate synthetic binary classification dataset."""
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    y = (X[:, :n_features//2].sum(axis=1) > X[:, n_features//2:].sum(axis=1)).astype(float) * 2 - 1
    return X, y


def train_qnn_lret_native_noise(device_name, device, n_qubits, n_layers, noise_rate, X_train, y_train, epochs, lr, log_file):
    """Train QNN using LRET's native noise model (no PennyLane noise channels)."""
    
    @qml.qnode(device)
    def qnn_circuit(params, x):
        # Data embedding layer (LRET handles noise internally via noise_prob)
        for i in range(n_qubits):
            qml.RY(x[i % len(x)] * np.pi, wires=i)
        
        # Variational layers (LRET applies noise after each gate automatically)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
            
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        return qml.expval(qml.PauliZ(0))
    
    def cost(params):
        loss = 0.0
        for x, y in zip(X_train, y_train):
            pred = qnn_circuit(params, x)
            loss += (pred - y) ** 2
        return loss / len(X_train)
    
    np.random.seed(123)
    params = pnp.array(np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2)), requires_grad=True)
    
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    
    history = {'epoch': [], 'loss': [], 'accuracy': [], 'time': []}
    
    start_time = time.time()
    epoch_start = start_time
    
    print(f"\n{'='*70}")
    print(f"Training on {device_name} with {noise_rate*100:.0f}% noise (native)")
    print(f"{'='*70}")
    print(f"{'Epoch':^8} | {'Loss':^12} | {'Accuracy':^10} | {'Time (s)':^12} | {'ETA (h)':^10}")
    print(f"{'-'*70}")
    
    for epoch in range(epochs):
        params, loss = opt.step_and_cost(cost, params)
        
        predictions = np.array([qnn_circuit(params, x) for x in X_train])
        predictions_binary = np.sign(predictions)
        accuracy = np.mean(predictions_binary == y_train)
        
        epoch_time = time.time() - epoch_start
        epoch_start = time.time()
        
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining_epochs = epochs - epoch - 1
        eta_hours = (avg_epoch_time * remaining_epochs) / 3600
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{epoch:^8} | {float(loss):^12.6f} | {accuracy:^10.2%} | {epoch_time:^12.1f} | {eta_hours:^10.2f}")
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{device_name},{epoch},{float(loss)},{accuracy},{epoch_time},{eta_hours}\n")
        
        history['epoch'].append(epoch)
        history['loss'].append(float(loss))
        history['accuracy'].append(float(accuracy))
        history['time'].append(epoch_time)
    
    total_time = time.time() - start_time
    print(f"{'-'*70}")
    print(f"Total training time: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
    
    return params, history, total_time


def train_qnn_noisy(device_name, device, n_qubits, n_layers, noise_rate, X_train, y_train, epochs, lr, log_file):
    """Train QNN with noise using PennyLane's built-in noise channels."""
    
    @qml.qnode(device)
    def qnn_circuit(params, x):
        # Data embedding layer with noise
        for i in range(n_qubits):
            qml.RY(x[i % len(x)] * np.pi, wires=i)
            qml.DepolarizingChannel(noise_rate, wires=i)
        
        # Variational layers with noise
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.DepolarizingChannel(noise_rate, wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
                qml.DepolarizingChannel(noise_rate, wires=i)
            
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
                # Depolarizing on both qubits after CNOT
                qml.DepolarizingChannel(noise_rate, wires=i)
                qml.DepolarizingChannel(noise_rate, wires=i+1)
        
        return qml.expval(qml.PauliZ(0))
    
    def cost(params):
        loss = 0.0
        for x, y in zip(X_train, y_train):
            pred = qnn_circuit(params, x)
            loss += (pred - y) ** 2
        return loss / len(X_train)
    
    np.random.seed(123)
    params = pnp.array(np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2)), requires_grad=True)
    
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    
    history = {'epoch': [], 'loss': [], 'accuracy': [], 'time': []}
    
    start_time = time.time()
    epoch_start = start_time
    
    print(f"\n{'='*70}")
    print(f"Training on {device_name} with {noise_rate*100:.0f}% noise")
    print(f"{'='*70}")
    print(f"{'Epoch':^8} | {'Loss':^12} | {'Accuracy':^10} | {'Time (s)':^12} | {'ETA (h)':^10}")
    print(f"{'-'*70}")
    
    for epoch in range(epochs):
        params, loss = opt.step_and_cost(cost, params)
        
        predictions = np.array([qnn_circuit(params, x) for x in X_train])
        predictions_binary = np.sign(predictions)
        accuracy = np.mean(predictions_binary == y_train)
        
        epoch_time = time.time() - epoch_start
        epoch_start = time.time()
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining_epochs = epochs - epoch - 1
        eta_hours = (avg_epoch_time * remaining_epochs) / 3600
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{epoch:^8} | {float(loss):^12.6f} | {accuracy:^10.2%} | {epoch_time:^12.1f} | {eta_hours:^10.2f}")
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{device_name},{epoch},{float(loss)},{accuracy},{epoch_time},{eta_hours}\n")
        
        history['epoch'].append(epoch)
        history['loss'].append(float(loss))
        history['accuracy'].append(float(accuracy))
        history['time'].append(epoch_time)
    
    total_time = time.time() - start_time
    print(f"{'-'*70}")
    print(f"Total training time: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
    
    return params, history, total_time


def run_benchmark(device_name, device_fn, n_qubits, n_layers, noise_rate, epochs, dataset_size, log_file):
    """Run complete benchmark for a device."""
    
    result = {
        'device': device_name,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'noise_rate': noise_rate,
        'epochs': epochs,
        'dataset_size': dataset_size,
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'error': None
    }
    
    try:
        print(f"\n{'#'*80}")
        print(f"# Device: {device_name}")
        print(f"# Qubits: {n_qubits}, Layers: {n_layers}, Noise: {noise_rate*100:.0f}%")
        print(f"# Epochs: {epochs}, Samples: {dataset_size}")
        print(f"{'#'*80}")
        
        print("\nGenerating dataset...")
        X, y = generate_dataset(dataset_size, n_qubits)
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        print(f"\nCreating {device_name} device...")
        start = time.time()
        dev = device_fn(wires=n_qubits)
        device_create_time = time.time() - start
        print(f"Device created in {device_create_time:.3f}s")
        result['device_create_time'] = device_create_time
        
        print("\nStarting training (this will take hours)...")
        print(f"Started at: {datetime.now().isoformat()}")
        
        # Use native noise for LRET, PennyLane noise channels for default.mixed
        if 'LRET' in device_name:
            trained_params, history, total_time = train_qnn_lret_native_noise(
                device_name, dev, n_qubits, n_layers, noise_rate, X, y, epochs, LEARNING_RATE, log_file
            )
        else:
            trained_params, history, total_time = train_qnn_noisy(
                device_name, dev, n_qubits, n_layers, noise_rate, X, y, epochs, LEARNING_RATE, log_file
            )
        
        result['status'] = 'success'
        result['total_training_time_seconds'] = total_time
        result['total_training_time_hours'] = total_time / 3600
        result['final_loss'] = history['loss'][-1]
        result['final_accuracy'] = history['accuracy'][-1]
        result['avg_epoch_time'] = np.mean(history['time'])
        result['history'] = history
        
        print(f"\n[OK] {device_name} completed successfully!")
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Final Accuracy: {result['final_accuracy']:.2%}")
        print(f"  Total Time: {result['total_training_time_hours']:.2f} hours")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"\n[FAIL] {device_name} FAILED: {e}")
        print(traceback.format_exc())
    
    return result


def main():
    print("="*80)
    print("NOISY QNN BENCHMARK - LRET vs default.mixed (FAIR COMPARISON)")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"\nParameters:")
    print(f"  - Qubits: {N_QUBITS}")
    print(f"  - Epochs: {N_EPOCHS}")
    print(f"  - Dataset size: {DATASET_SIZE}")
    print(f"  - Layers: {N_LAYERS}")
    print(f"  - Noise rate: {NOISE_RATE*100:.0f}%")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"\nExpected runtime: Several hours (default.mixed is very slow)")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f'noisy_qnn_10q_log_{timestamp}.csv')
    with open(log_file, 'w') as f:
        f.write("timestamp,device,epoch,loss,accuracy,epoch_time,eta_hours\n")
    print(f"Logging to: {log_file}")
    
    results = []
    
    # LRET first (should be faster)
    devices = [
        ("LRET (Native C++)", lambda wires: QLRETDevice(wires=wires, noise_prob=NOISE_RATE)),
        ("default.mixed", lambda wires: qml.device("default.mixed", wires=wires)),
    ]
    
    for device_name, device_fn in devices:
        result = run_benchmark(
            device_name=device_name,
            device_fn=device_fn,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            noise_rate=NOISE_RATE,
            epochs=N_EPOCHS,
            dataset_size=DATASET_SIZE,
            log_file=log_file
        )
        results.append(result)
        
        results_file = os.path.join(RESULTS_DIR, f'noisy_qnn_10q_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json_results = []
            for r in results:
                r_copy = r.copy()
                if 'history' in r_copy and r_copy['history']:
                    r_copy['history'] = {k: [float(v) for v in vals] for k, vals in r_copy['history'].items()}
                json_results.append(r_copy)
            json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    
    if len(successful) >= 2:
        lret = next((r for r in successful if 'LRET' in r['device']), None)
        mixed = next((r for r in successful if 'mixed' in r['device']), None)
        
        if lret and mixed:
            speedup = mixed['total_training_time_seconds'] / lret['total_training_time_seconds']
            print(f"\n*** SPEEDUP: {speedup:.1f}x {'(LRET faster!)' if speedup > 1 else '(default.mixed faster)'} ***")
            print(f"\nLRET:")
            print(f"  Time: {lret['total_training_time_hours']:.2f} hours")
            print(f"  Final Accuracy: {lret['final_accuracy']:.2%}")
            print(f"\ndefault.mixed:")
            print(f"  Time: {mixed['total_training_time_hours']:.2f} hours")
            print(f"  Final Accuracy: {mixed['final_accuracy']:.2%}")
    
    for r in results:
        print(f"\n{r['device']}:")
        print(f"  Status: {r['status']}")
        if r['status'] == 'success':
            print(f"  Training Time: {r['total_training_time_hours']:.2f} hours")
            print(f"  Final Accuracy: {r['final_accuracy']:.2%}")
        else:
            print(f"  Error: {r.get('error', 'Unknown')}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {results_file}")
    print(f"End time: {datetime.now().isoformat()}")
    print("="*80)


if __name__ == "__main__":
    main()
