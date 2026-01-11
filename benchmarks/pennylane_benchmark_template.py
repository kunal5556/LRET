#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              PENNYLANE BENCHMARK TEMPLATE FOR LRET SIMULATOR                 ║
║                                                                              ║
║  Purpose: Comprehensive benchmarking script for comparing LRET quantum      ║
║           simulator against PennyLane's built-in devices (default.mixed,    ║
║           lightning.qubit, etc.) for variational quantum algorithms.         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
                            SETUP GUIDE FOR NEW SYSTEMS
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: If you cloned this repository on a NEW system, you MUST rebuild 
the LRET C++ backend before running this benchmark. The compiled native 
module (.pyd/.so) is platform-specific and cannot be reused across systems.

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Install System Dependencies                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Windows:
    1. Install Visual Studio 2019/2022 with "Desktop development with C++"
    2. Install CMake 3.16+: https://cmake.org/download/
    3. Install Python 3.8+ with dev headers (already included in standard install)

Linux (Ubuntu/Debian):
    sudo apt-get update
    sudo apt-get install build-essential cmake python3-dev libeigen3-dev

macOS:
    brew install cmake eigen python@3.11

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Install Python Dependencies                                        │
└─────────────────────────────────────────────────────────────────────────────┘

    pip install pennylane numpy scipy psutil

Optional (for specific comparisons):
    pip install torch  # For PyTorch integration
    pip install jax jaxlib  # For JAX integration

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Build LRET C++ Backend                                             │
└─────────────────────────────────────────────────────────────────────────────┘

From the LRET repository root:

    # Create build directory
    mkdir build
    cd build
    
    # Configure CMake (auto-detects compiler and Python)
    cmake .. -DUSE_PYTHON=ON
    
    # Build (adjust -j flag based on CPU cores)
    # Windows:
    cmake --build . --config Release
    
    # Linux/macOS:
    make -j$(nproc)  # Linux
    make -j$(sysctl -n hw.ncpu)  # macOS

Expected output: You should see "_qlret_native.pyd" (Windows) or 
"_qlret_native.so" (Linux/macOS) in the build directory.

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Install LRET Python Package                                        │
└─────────────────────────────────────────────────────────────────────────────┘

From the LRET repository root:

    cd python
    pip install -e .

This registers LRET as a PennyLane device plugin.

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Verify Installation                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Test that LRET device is available:

    python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print('✓ LRET device loaded:', dev.name)"

Expected output: "✓ LRET device loaded: QLRET Simulator"

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Run This Benchmark Script                                          │
└─────────────────────────────────────────────────────────────────────────────┘

    python benchmarks/pennylane_benchmark_template.py

Results will be saved to: results/benchmark_<timestamp>/

═══════════════════════════════════════════════════════════════════════════════
                           TROUBLESHOOTING COMMON ISSUES
═══════════════════════════════════════════════════════════════════════════════

Issue 1: "ModuleNotFoundError: No module named '_qlret_native'"
    Solution: Rebuild the C++ backend (Step 3 above)
    
Issue 2: "Device 'qlret.mixed' not found"
    Solution: Reinstall Python package: cd python && pip install -e . --force-reinstall
    
Issue 3: CMake cannot find Python
    Solution: Specify Python explicitly: cmake .. -DPYTHON_EXECUTABLE=/path/to/python
    
Issue 4: "Kraus operator not supported"
    Solution: Ensure you pulled latest code with Kraus support (commit 8ebbfaa+)
    
Issue 5: Benchmark runs slowly on Windows
    Solution: Close unnecessary programs, check Windows Defender isn't scanning

═══════════════════════════════════════════════════════════════════════════════
"""

import time
import sys
import os
import json
import numpy as np
import pennylane as qml
import psutil
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     BENCHMARK CONFIGURATION SECTION                       ║
# ║                                                                           ║
# ║  Modify parameters below to customize your benchmark. Detailed comments  ║
# ║  explain each option and how it affects the benchmark.                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM CIRCUIT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Number of qubits in the quantum circuit
# Options: 2-24 (practical range)
# - 2-4 qubits: Fast, good for debugging (~seconds to minutes)
# - 5-8 qubits: Moderate, realistic for NISQ devices (~minutes to hours)
# - 9-12 qubits: Slow, tests memory efficiency (~hours to days)
# - 13+ qubits: Very slow, only for scalability testing
# Recommendation: Start with 4, increase to test limits
N_QUBITS = 4

# Number of variational layers in the ansatz
# Options: 1-10
# - 1 layer: Shallow circuit, may underfit
# - 2-3 layers: Good balance for small problems
# - 4-6 layers: Deeper expressibility, longer runtime
# - 7+ layers: Risk of barren plateaus
# Each layer adds: N_QUBITS × 2 parameters + (N_QUBITS-1) CNOTs
N_VARIATIONAL_LAYERS = 2

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Number of training epochs
# Options: 1-1000
# - 1-10: Quick test
# - 50-100: Standard convergence testing
# - 200+: Deep convergence analysis
# Note: More epochs = longer runtime (linear scaling)
N_EPOCHS = 100

# Batch size (number of samples per epoch)
# Options: 1-100
# - 1-10: Small batch, noisy gradients
# - 10-30: Medium batch, good balance
# - 50-100: Large batch, smooth gradients but slower per epoch
# Each sample requires: 1 + (N_PARAMS × 2) circuit evaluations for numerical gradient
N_SAMPLES = 25

# Learning rate for gradient descent
# Options: 0.001 - 1.0
# - 0.001-0.01: Conservative, slow convergence
# - 0.05-0.2: Standard, good convergence
# - 0.5-1.0: Aggressive, may oscillate
# Affects how fast parameters update during training
LEARNING_RATE = 0.1

# Random seed for reproducibility
# Options: Any integer, or None for random
# - Fixed integer (e.g., 42): Reproducible results
# - None: Different results each run
# Same seed = same training data, initialization, and results
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# NOISE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Whether to add noise to the circuit
# Options: True or False
# - True: Realistic NISQ simulation (requires mixed-state simulator)
# - False: Ideal noiseless simulation (can use state vector)
USE_NOISE = True

# Noise type to apply (only if USE_NOISE=True)
# Options:
#   "depolarizing"   - Uniform noise on all Pauli channels (most common)
#   "amplitude"      - T1 decay (energy relaxation)
#   "phase"          - T2 dephasing (phase randomization)
#   "bitflip"        - X errors only
#   "phaseflip"      - Z errors only
# Recommendation: "depolarizing" for general benchmarking
NOISE_TYPE = "depolarizing"

# Noise rate (error probability per qubit)
# Options: 0.0 - 1.0
# - 0.0: No noise (USE_NOISE should be False)
# - 0.001-0.02: Realistic NISQ devices (IBM, Google)
# - 0.05-0.15: Moderate noise for testing
# - 0.2+: High noise, challenging regime
# Higher noise = more challenging for simulator
NOISE_RATE = 0.10

# Where to apply noise
# Options:
#   "encoding"  - Only after data encoding layer
#   "all"       - After every gate (realistic but slow)
#   "layers"    - After each variational layer
# Recommendation: "encoding" for speed, "all" for realism
NOISE_LOCATION = "encoding"

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Which devices to benchmark (list of device names)
# Options:
#   "qlret.mixed"       - LRET simulator (C++ backend, low-rank)
#   "default.mixed"     - PennyLane density matrix (Python/NumPy)
#   "lightning.qubit"   - PennyLane state vector (C++, no noise support)
#   "default.qubit"     - PennyLane state vector (Python, no noise support)
#
# Note: For noisy circuits (USE_NOISE=True), only mixed-state devices work:
#   - "qlret.mixed" ✓
#   - "default.mixed" ✓
#   - "lightning.qubit" ✗ (will skip with warning)
#   - "default.qubit" ✗ (will skip with warning)
#
# Recommendation: ["qlret.mixed", "default.mixed"] for noisy comparison
DEVICES_TO_TEST = ["qlret.mixed", "default.mixed"]

# LRET-specific configuration (only used if "qlret.mixed" in DEVICES_TO_TEST)
LRET_CONFIG = {
    "epsilon": 1e-4,  # Truncation threshold for low-rank approximation
                      # Options: 1e-3 (faster, less accurate) to 1e-6 (slower, more accurate)
}

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT COMPUTATION METHOD
# ─────────────────────────────────────────────────────────────────────────────

# How to compute gradients
# Options:
#   "numerical"         - Finite differences (slow but universal)
#   "parameter_shift"   - Parameter-shift rule (exact for quantum gates)
#   "adjoint"           - Adjoint differentiation (fast but limited support)
#
# Notes:
# - "numerical": Works with all devices, fair comparison
# - "parameter_shift": More efficient but device-dependent
# - "adjoint": Fastest but not supported by all devices
# Recommendation: "numerical" for fair benchmarking
GRADIENT_METHOD = "numerical"

# Finite difference shift (only for GRADIENT_METHOD="numerical")
# Options: 0.01 - 0.5
# - 0.01: High precision, more stable
# - 0.1: Standard, good balance
# - 0.5: Low precision, faster
FINITE_DIFF_SHIFT = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT ARCHITECTURE OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Gate types for single-qubit rotations in variational layers
# Options: ["RY", "RZ"], ["RX", "RY"], ["RX", "RZ"], ["RY", "RZ", "RX"]
# - 2 gates: Faster, less expressive
# - 3 gates: Universal single-qubit rotation
# Standard: ["RY", "RZ"] (efficient and universal)
ROTATION_GATES = ["RY", "RZ"]

# Entanglement pattern for two-qubit gates
# Options:
#   "linear"    - Chain: 0→1, 1→2, 2→3, ... (N-1 CNOTs per layer)
#   "circular"  - Ring: linear + last→first (N CNOTs per layer)
#   "full"      - All pairs: expensive but maximally entangling
# Recommendation: "linear" for speed, "circular" for periodic systems
ENTANGLEMENT = "linear"

# Two-qubit gate type
# Options: "CNOT", "CZ", "CY", "SWAP"
# - CNOT: Standard, most common
# - CZ: Phase-flip controlled gate
# - CY: Y-controlled gate
# - SWAP: Exchanges qubit states
# Recommendation: "CNOT" (universal and well-supported)
TWO_QUBIT_GATE = "CNOT"

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Type of classification problem
# Options:
#   "linear"     - Linearly separable (easy, sum of features)
#   "xor"        - XOR problem (harder, non-linear)
#   "circle"     - Circular decision boundary
#   "random"     - Random labels (no pattern, tests convergence limits)
# Recommendation: "linear" for standard benchmarking
PROBLEM_TYPE = "linear"

# Data normalization
# Options: True or False
# - True: Scale features to [-1, 1] or [0, 1] (recommended)
# - False: Use raw random data
# Normalization helps training stability
NORMALIZE_DATA = True

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT AND LOGGING
# ─────────────────────────────────────────────────────────────────────────────

# How often to log progress (in epochs)
# Options: 1 - N_EPOCHS
# - 1: Log every epoch (verbose, large logs)
# - 10: Log every 10 epochs (moderate)
# - 100: Log milestones only (minimal)
# Recommendation: 1 for debugging, 5-10 for production
LOG_FREQUENCY = 1

# Whether to save epoch-by-epoch data to CSV
# Options: True or False
# - True: Save detailed CSV files (useful for plotting)
# - False: Only save final results (smaller output)
SAVE_EPOCH_DATA = True

# Whether to calculate and log additional metrics
# Options: True or False
# - True: Calculate fidelity, purity, accuracy, etc. (slower)
# - False: Only calculate loss (faster)
CALCULATE_METRICS = False

# Output directory (relative to script location)
# Options: Any valid directory path
# Results will be saved to: OUTPUT_DIR/benchmark_<timestamp>/
OUTPUT_DIR = "results"

# ═════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    END OF CONFIGURATION SECTION                           ║
# ║                                                                           ║
# ║  Code below this line should not need modification for standard usage.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ═════════════════════════════════════════════════════════════════════════════

# =============================================================================
# SETUP LOGGING AND OUTPUT
# =============================================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR, f"benchmark_{RUN_ID}")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "benchmark.log")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.log")
RESULTS_FILE = os.path.join(LOG_DIR, "results.json")
CONFIG_FILE = os.path.join(LOG_DIR, "config.json")

def log(msg: str, progress: bool = False, console: bool = True):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if console:
        print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    if progress:
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

def save_results(data: Dict[str, Any]):
    """Save results to JSON"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def save_epoch_data(filename: str, epoch: int, loss: float, time_s: float):
    """Append epoch data to CSV"""
    if not SAVE_EPOCH_DATA:
        return
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("epoch,loss,time_seconds\n")
    with open(filename, 'a') as f:
        f.write(f"{epoch},{loss:.10f},{time_s:.6f}\n")

# =============================================================================
# CONFIGURATION VALIDATION AND SAVING
# =============================================================================

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if N_QUBITS < 2 or N_QUBITS > 24:
        errors.append(f"N_QUBITS={N_QUBITS} is out of practical range [2, 24]")
    
    if USE_NOISE and not any(dev in ["qlret.mixed", "default.mixed"] for dev in DEVICES_TO_TEST):
        errors.append("USE_NOISE=True requires at least one mixed-state device (qlret.mixed or default.mixed)")
    
    if NOISE_RATE < 0 or NOISE_RATE > 1:
        errors.append(f"NOISE_RATE={NOISE_RATE} must be in [0.0, 1.0]")
    
    if N_SAMPLES < 1:
        errors.append(f"N_SAMPLES={N_SAMPLES} must be at least 1")
    
    if N_EPOCHS < 1:
        errors.append(f"N_EPOCHS={N_EPOCHS} must be at least 1")
    
    if errors:
        log("Configuration validation failed:")
        for err in errors:
            log(f"  ERROR: {err}")
        sys.exit(1)

config_dict = {
    "run_id": RUN_ID,
    "quantum_circuit": {
        "n_qubits": N_QUBITS,
        "n_variational_layers": N_VARIATIONAL_LAYERS,
        "rotation_gates": ROTATION_GATES,
        "entanglement": ENTANGLEMENT,
        "two_qubit_gate": TWO_QUBIT_GATE,
    },
    "training": {
        "n_epochs": N_EPOCHS,
        "n_samples": N_SAMPLES,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "gradient_method": GRADIENT_METHOD,
        "finite_diff_shift": FINITE_DIFF_SHIFT if GRADIENT_METHOD == "numerical" else None,
    },
    "noise": {
        "use_noise": USE_NOISE,
        "noise_type": NOISE_TYPE if USE_NOISE else None,
        "noise_rate": NOISE_RATE if USE_NOISE else 0.0,
        "noise_location": NOISE_LOCATION if USE_NOISE else None,
    },
    "devices": {
        "devices_to_test": DEVICES_TO_TEST,
        "lret_config": LRET_CONFIG if "qlret.mixed" in DEVICES_TO_TEST else None,
    },
    "data": {
        "problem_type": PROBLEM_TYPE,
        "normalize_data": NORMALIZE_DATA,
    },
    "output": {
        "log_frequency": LOG_FREQUENCY,
        "save_epoch_data": SAVE_EPOCH_DATA,
        "calculate_metrics": CALCULATE_METRICS,
        "output_dir": LOG_DIR,
    },
    "system": {
        "python_version": sys.version.split()[0],
        "pennylane_version": qml.__version__,
        "numpy_version": np.__version__,
    },
}

# Save configuration
with open(CONFIG_FILE, 'w') as f:
    json.dump(config_dict, f, indent=2)

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_training_data(n_samples: int, n_features: int, problem_type: str, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data based on problem type"""
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    
    if NORMALIZE_DATA:
        X = X / (np.max(np.abs(X)) + 1e-8)
    
    if problem_type == "linear":
        # Linear separability: sign of sum of first 2 features
        y = np.sign(np.sum(X[:, :min(2, n_features)], axis=1))
    elif problem_type == "xor":
        # XOR: product of signs of first 2 features
        y = np.sign(X[:, 0]) * np.sign(X[:, 1]) if n_features >= 2 else np.ones(n_samples)
    elif problem_type == "circle":
        # Circular boundary
        y = np.where(np.sum(X**2, axis=1) < 0.5, 1.0, -1.0)
    elif problem_type == "random":
        # Random labels (no pattern)
        y = np.random.choice([-1, 1], size=n_samples).astype(np.float64)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    
    # Ensure no zeros (replace with 1.0)
    y = np.where(y == 0, 1.0, y)
    
    return X, y

# =============================================================================
# NOISE CHANNEL HELPER
# =============================================================================

def apply_noise_channel(qubits: List[int], noise_type: str, noise_rate: float):
    """Apply noise channel to specified qubits"""
    if not USE_NOISE or noise_rate == 0:
        return
    
    for q in qubits:
        if noise_type == "depolarizing":
            qml.DepolarizingChannel(noise_rate, wires=q)
        elif noise_type == "amplitude":
            qml.AmplitudeDamping(noise_rate, wires=q)
        elif noise_type == "phase":
            qml.PhaseDamping(noise_rate, wires=q)
        elif noise_type == "bitflip":
            qml.BitFlip(noise_rate, wires=q)
        elif noise_type == "phaseflip":
            qml.PhaseFlip(noise_rate, wires=q)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

# =============================================================================
# CIRCUIT CONSTRUCTION
# =============================================================================

def make_circuit(dev, n_qubits: int, n_layers: int):
    """Create parameterized quantum circuit"""
    
    @qml.qnode(dev)
    def circuit(params, x):
        # Data encoding layer
        for i in range(n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
        
        # Apply noise after encoding if configured
        if NOISE_LOCATION == "encoding":
            apply_noise_channel(list(range(n_qubits)), NOISE_TYPE, NOISE_RATE)
        
        # Variational layers
        for layer in range(n_layers):
            # Single-qubit rotations
            for i in range(n_qubits):
                param_idx = 0
                for gate_type in ROTATION_GATES:
                    angle = params[layer, i, param_idx]
                    if gate_type == "RX":
                        qml.RX(angle, wires=i)
                    elif gate_type == "RY":
                        qml.RY(angle, wires=i)
                    elif gate_type == "RZ":
                        qml.RZ(angle, wires=i)
                    param_idx += 1
            
            # Two-qubit entangling gates
            if ENTANGLEMENT == "linear":
                pairs = [(i, i+1) for i in range(n_qubits - 1)]
            elif ENTANGLEMENT == "circular":
                pairs = [(i, (i+1) % n_qubits) for i in range(n_qubits)]
            elif ENTANGLEMENT == "full":
                pairs = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
            else:
                raise ValueError(f"Unknown entanglement: {ENTANGLEMENT}")
            
            for ctrl, targ in pairs:
                if TWO_QUBIT_GATE == "CNOT":
                    qml.CNOT(wires=[ctrl, targ])
                elif TWO_QUBIT_GATE == "CZ":
                    qml.CZ(wires=[ctrl, targ])
                elif TWO_QUBIT_GATE == "CY":
                    qml.CY(wires=[ctrl, targ])
                elif TWO_QUBIT_GATE == "SWAP":
                    qml.SWAP(wires=[ctrl, targ])
            
            # Apply noise after layer if configured
            if NOISE_LOCATION in ["all", "layers"]:
                apply_noise_channel(list(range(n_qubits)), NOISE_TYPE, NOISE_RATE)
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    return circuit

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_circuit(name: str, circuit, init_params: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, csv_file: str) -> Dict[str, Any]:
    """Train quantum circuit with gradient descent"""
    
    log("=" * 70, progress=True)
    log(f"TRAINING: {name}", progress=True)
    log("=" * 70, progress=True)
    
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss / (1024**2)
    
    params = init_params.copy()
    losses = []
    epoch_times = []
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for x, target in zip(X_train, y_train):
            # Forward pass
            pred = float(circuit(params, x))
            loss = (pred - target) ** 2
            epoch_loss += loss
            
            # Gradient computation (numerical finite differences)
            if GRADIENT_METHOD == "numerical":
                grad = np.zeros_like(params)
                shift = FINITE_DIFF_SHIFT
                for idx in np.ndindex(params.shape):
                    p_plus = params.copy()
                    p_plus[idx] += shift
                    p_minus = params.copy()
                    p_minus[idx] -= shift
                    grad[idx] = (circuit(p_plus, x) - circuit(p_minus, x)) / (2 * shift)
                
                # Update parameters
                params = params - LEARNING_RATE * 2 * (pred - target) * grad
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / N_SAMPLES
        losses.append(float(avg_loss))
        epoch_times.append(epoch_time)
        
        # Save epoch data
        save_epoch_data(csv_file, epoch + 1, avg_loss, epoch_time)
        
        # Log progress
        if (epoch + 1) % LOG_FREQUENCY == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
            log(f"  [{name}] Epoch {epoch+1:3d}/{N_EPOCHS}: loss={avg_loss:.6f}, "
                f"time={epoch_time:.1f}s, elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min", 
                progress=True)
    
    # Final statistics
    total_time = time.time() - start_time
    end_mem = psutil.Process().memory_info().rss / (1024**2)
    
    result = {
        "status": "completed",
        "total_time_seconds": total_time,
        "avg_epoch_time": total_time / N_EPOCHS,
        "memory_start_mb": start_mem,
        "memory_end_mb": end_mem,
        "memory_delta_mb": end_mem - start_mem,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "min_loss": min(losses),
        "losses": losses,
        "epoch_times": epoch_times,
    }
    
    log(f"  [{name}] COMPLETED: {total_time:.1f}s total, "
        f"{total_time/N_EPOCHS:.1f}s/epoch, final_loss={losses[-1]:.6f}", progress=True)
    log("", progress=True)
    
    return result

# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main():
    log("=" * 70)
    log("PENNYLANE BENCHMARK - LRET vs PennyLane Devices")
    log("=" * 70)
    log(f"Run ID: {RUN_ID}")
    log(f"Output directory: {LOG_DIR}")
    log(f"PennyLane version: {qml.__version__}")
    log(f"Python version: {sys.version.split()[0]}")
    log("")
    
    # Validate configuration
    validate_config()
    
    # Log configuration
    log("CONFIGURATION:")
    log(f"  Qubits:            {N_QUBITS}")
    log(f"  Variational layers:{N_VARIATIONAL_LAYERS}")
    log(f"  Epochs:            {N_EPOCHS}")
    log(f"  Batch size:        {N_SAMPLES}")
    log(f"  Noise:             {NOISE_TYPE if USE_NOISE else 'None'} ({NOISE_RATE:.0%})")
    log(f"  Learning rate:     {LEARNING_RATE}")
    log(f"  Random seed:       {RANDOM_SEED}")
    log(f"  Devices:           {', '.join(DEVICES_TO_TEST)}")
    log("")
    
    # Calculate parameter count
    n_params_per_qubit = len(ROTATION_GATES)
    total_params = N_VARIATIONAL_LAYERS * N_QUBITS * n_params_per_qubit
    log(f"Total parameters: {total_params} ({N_VARIATIONAL_LAYERS} layers × {N_QUBITS} qubits × {n_params_per_qubit} angles)")
    log("")
    
    # Initialize results
    results = {
        "run_id": RUN_ID,
        "config": config_dict,
        "status": "initializing",
        "devices": {},
    }
    save_results(results)
    
    # Generate training data
    log("Generating training data...")
    X_train, y_train = generate_training_data(N_SAMPLES, N_QUBITS, PROBLEM_TYPE, RANDOM_SEED)
    log(f"  Training set: {N_SAMPLES} samples × {N_QUBITS} features")
    log(f"  Label distribution: {np.sum(y_train == 1)} positive, {np.sum(y_train == -1)} negative")
    log("")
    
    # Initialize parameters
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
    init_params = np.random.randn(N_VARIATIONAL_LAYERS, N_QUBITS, n_params_per_qubit) * 0.1
    log(f"Initialized parameters with shape: {init_params.shape}")
    log("")
    
    # Create devices and run benchmarks
    log("Creating devices and running benchmarks...")
    log("")
    
    for device_name in DEVICES_TO_TEST:
        try:
            # Check if device supports noise
            if USE_NOISE and device_name not in ["qlret.mixed", "default.mixed"]:
                log(f"  Skipping {device_name}: does not support noise channels")
                results["devices"][device_name] = {"status": "skipped", "reason": "no_noise_support"}
                continue
            
            # Create device
            if device_name == "qlret.mixed":
                dev = qml.device(device_name, wires=N_QUBITS, **LRET_CONFIG)
            else:
                dev = qml.device(device_name, wires=N_QUBITS)
            
            log(f"  Created device: {dev.name}")
            
            # Create circuit
            circuit = make_circuit(dev, N_QUBITS, N_VARIATIONAL_LAYERS)
            
            # Train
            csv_file = os.path.join(LOG_DIR, f"{device_name.replace('.', '_')}_epochs.csv")
            device_result = train_circuit(device_name, circuit, init_params, X_train, y_train, csv_file)
            
            results["devices"][device_name] = device_result
            results["status"] = f"running_{len(results['devices'])}/{len(DEVICES_TO_TEST)}"
            save_results(results)
            
        except Exception as e:
            log(f"  ERROR with {device_name}: {e}")
            log(traceback.format_exc())
            results["devices"][device_name] = {"status": "failed", "error": str(e)}
            save_results(results)
    
    # Generate summary
    log("")
    log("=" * 70)
    log("BENCHMARK COMPLETE")
    log("=" * 70)
    log("")
    
    results["status"] = "completed"
    
    # Compare devices
    completed_devices = {k: v for k, v in results["devices"].items() if v.get("status") == "completed"}
    
    if len(completed_devices) >= 2:
        device_names = list(completed_devices.keys())
        device1, device2 = device_names[0], device_names[1]
        
        time1 = completed_devices[device1]["total_time_seconds"]
        time2 = completed_devices[device2]["total_time_seconds"]
        speedup = time2 / time1 if time1 < time2 else time1 / time2
        faster = device1 if time1 < time2 else device2
        
        loss1 = completed_devices[device1]["final_loss"]
        loss2 = completed_devices[device2]["final_loss"]
        loss_diff = abs(loss1 - loss2)
        
        log(f"{'METRIC':<30} {device1:>20} {device2:>20} {'RATIO':>12}")
        log("-" * 85)
        log(f"{'Total time (seconds)':<30} {time1:>20.1f} {time2:>20.1f} {speedup:>11.2f}x")
        log(f"{'Avg time per epoch (s)':<30} {time1/N_EPOCHS:>20.1f} {time2/N_EPOCHS:>20.1f}")
        log(f"{'Final loss':<30} {loss1:>20.6f} {loss2:>20.6f}")
        log(f"{'Loss difference':<30} {loss_diff:>20.6f}")
        log("")
        
        log(f"[RESULT] {faster} is {speedup:.2f}x FASTER")
        log(f"[RESULT] Results {'MATCH' if loss_diff < 0.01 else 'DIFFER'} (difference: {loss_diff:.6f})")
        
        results["summary"] = {
            "speedup": speedup,
            "faster_device": faster,
            "loss_difference": loss_diff,
            "results_match": loss_diff < 0.01,
        }
    
    save_results(results)
    
    log("")
    log("=" * 70)
    log("OUTPUT FILES:")
    log(f"  Configuration:  {CONFIG_FILE}")
    log(f"  Main log:       {LOG_FILE}")
    log(f"  Progress log:   {PROGRESS_FILE}")
    log(f"  Results JSON:   {RESULTS_FILE}")
    if SAVE_EPOCH_DATA:
        for device_name in completed_devices:
            csv_name = device_name.replace('.', '_')
            log(f"  {device_name} epochs: {csv_name}_epochs.csv")
    log("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)
