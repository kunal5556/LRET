# Phase 9.3: Adaptive & ML-Driven QEC - Detailed Implementation Strategy

**Status:** Planned  
**Timeline:** 2-3 weeks  
**Model:** Claude Sonnet 4.5 (strategy), Claude Opus 4.5 (implementation)  
**Prerequisites:** Phase 9.1 (QEC foundation) ✅ | Phase 9.2 (Distributed QEC) ✅  
**Dependencies:** Phase 4 (Noise models) | Phase 5 (Python bindings) | Phase 8 (Fault tolerance)

---

## 1. Executive Summary

**Objective:** Implement runtime-adaptive quantum error correction that dynamically selects optimal codes and uses ML-trained decoders based on real-time noise profiles.

**Key Innovations:**
1. **Noise-Adaptive Code Selection:** Automatically choose code type (Surface, Repetition, Color) based on calibrated T1/T2/gate error profiles
2. **ML-Based Decoder:** Neural network decoder trained on synthetic syndrome data, outperforms MWPM on correlated noise
3. **Closed-Loop Calibration:** Continuous feedback from QEC performance back to noise model updates
4. **Dynamic Distance Scaling:** Adjust code distance based on real-time logical error rate monitoring

**Success Metrics:**
- Adaptive selector improves logical fidelity by ≥10% vs fixed Surface code
- ML decoder achieves ≥95% accuracy on test syndromes (vs 90% MWPM baseline on correlated noise)
- Closed-loop maintains p_L < threshold under 20% noise drift
- Distance scaling prevents logical errors during noise bursts

---

## 2. Architecture Overview

### 2.1 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                  AdaptiveQECController                       │
│  - Orchestrates code selection, decoding, calibration       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌───────────────┐ ┌──────────┐ ┌────────────────┐
│ CodeSelector  │ │ MLDecoder│ │ CalibrationLoop│
│ (C++)         │ │ (C++/Py) │ │ (C++ + Phase 4)│
└───────────────┘ └──────────┘ └────────────────┘
        │             │             │
        │             │             │
        ▼             ▼             ▼
┌───────────────┐ ┌──────────┐ ┌────────────────┐
│ NoiseProfile  │ │ TrainData│ │ DriftDetector  │
│ (from Phase 4)│ │ Generator│ │ (Stats-based)  │
└───────────────┘ └──────────┘ └────────────────┘
```

### 2.2 Data Flow

```
1. Calibration → NoiseProfile (Phase 4.2)
2. NoiseProfile → AdaptiveCodeSelector → QECCodeType
3. QECCodeType → DistributedLogicalQubit (Phase 9.2)
4. Syndrome → MLDecoder → Correction
5. Runtime Stats → DriftDetector → Recalibration Trigger
6. Loop back to step 1
```

---

## 3. Detailed Component Specifications

### 3.1 NoiseProfile Data Structure

**Purpose:** Unified representation of device noise characteristics from Phase 4 calibration.

**Header:** `include/qec_adaptive.h`

```cpp
namespace qlret {

// Noise profile aggregated from Phase 4 calibration
struct NoiseProfile {
    // Single-qubit coherence times (per qubit)
    std::vector<double> t1_times_ns;       // T1 for each qubit
    std::vector<double> t2_times_ns;       // T2 for each qubit
    
    // Single-qubit gate errors (averaged across H, X, RZ, etc.)
    std::vector<double> single_gate_errors;  // Error probability per qubit
    
    // Two-qubit gate errors (CNOT/CZ)
    std::map<std::pair<size_t, size_t>, double> two_qubit_errors;
    
    // Measurement errors
    std::vector<double> readout_errors;      // P(1|0) + P(0|1)
    
    // Advanced noise features
    std::vector<CorrelatedError> correlated_errors;  // From Phase 4.3
    TimeVaryingNoiseParams time_varying;             // From Phase 4.3
    std::vector<MemoryEffect> memory_effects;        // From Phase 4.3
    
    // Derived statistics
    double avg_gate_error() const;
    double max_gate_error() const;
    double t1_t2_ratio() const;  // Indicates bias direction
    bool is_biased() const;      // |T1/T2 - 1| > 0.2
    bool has_correlations() const;
    
    // Serialization
    nlohmann::json to_json() const;
    static NoiseProfile from_json(const nlohmann::json& j);
    
    // Load from Phase 4 calibration output
    static NoiseProfile load_from_calibration(const std::string& calib_file);
};

} // namespace qlret
```

**Implementation Notes:**
- `load_from_calibration()` parses JSON from Phase 4.2 scripts (`calibrate_noise_model.py` output)
- `is_biased()` threshold: 0.2 → prefer repetition codes for X or Z bias
- `has_correlations()` checks if any `correlated_errors` entries exist

**Test:** `test_noise_profile.cpp`
- Load sample JSON from `tests/fixtures/ibmq_noise_profile.json`
- Verify `avg_gate_error()` matches manual calculation
- Test `is_biased()` for T1=100µs, T2=50µs → true

---

### 3.2 AdaptiveCodeSelector

**Purpose:** Select optimal QEC code type based on noise characteristics.

**Header:** `include/qec_adaptive.h`

```cpp
namespace qlret {

class AdaptiveCodeSelector {
public:
    struct Config {
        double bias_threshold = 0.2;         // T1/T2 ratio threshold for bias detection
        double high_error_threshold = 0.05;  // Switch to higher distance or concat codes
        bool prefer_low_overhead = true;     // Minimize qubit count when possible
    };
    
    explicit AdaptiveCodeSelector(Config config = {});
    
    // Main selection interface
    StabilizerCodeType select_code(const NoiseProfile& noise);
    
    // Decision tree for code selection
    StabilizerCodeType select_for_biased_noise(const NoiseProfile& noise);
    StabilizerCodeType select_for_correlated_noise(const NoiseProfile& noise);
    StabilizerCodeType select_for_high_error_rate(const NoiseProfile& noise);
    
    // Predict logical error rate for given code + noise combination
    double predict_logical_error_rate(
        StabilizerCodeType code,
        size_t distance,
        const NoiseProfile& noise
    );
    
    // Distance selection based on target logical error rate
    size_t select_distance(
        StabilizerCodeType code,
        const NoiseProfile& noise,
        double target_logical_error_rate
    );
    
    // Comparison: rank all codes for given noise profile
    std::vector<std::pair<StabilizerCodeType, double>> rank_codes(
        const NoiseProfile& noise,
        size_t distance
    );
    
private:
    Config config_;
    
    // Helper: compute effective error rate for code
    double compute_effective_error_rate(
        StabilizerCodeType code,
        const NoiseProfile& noise
    );
};

} // namespace qlret
```

**Selection Algorithm:**

```cpp
StabilizerCodeType AdaptiveCodeSelector::select_code(const NoiseProfile& noise) {
    // Decision tree based on noise characteristics
    
    // 1. High error rate → concatenated or high-distance codes
    if (noise.avg_gate_error() > config_.high_error_threshold) {
        return select_for_high_error_rate(noise);
    }
    
    // 2. Biased noise (T1 << T2 or T1 >> T2) → repetition or biased surface
    if (noise.is_biased()) {
        return select_for_biased_noise(noise);
    }
    
    // 3. Correlated errors → surface code (better at handling clusters)
    if (noise.has_correlations()) {
        return select_for_correlated_noise(noise);
    }
    
    // 4. Default: standard rotated surface code
    return StabilizerCodeType::SURFACE;
}
```

**Prediction Model:**

```cpp
double AdaptiveCodeSelector::predict_logical_error_rate(
    StabilizerCodeType code,
    size_t distance,
    const NoiseProfile& noise
) {
    double p_phys = compute_effective_error_rate(code, noise);
    
    // Simplified model: p_L ≈ A * p_phys^((d+1)/2)
    // For surface code: A ≈ 0.1, exponent ≈ (d+1)/2
    // For repetition: exponent ≈ d (bit-flip only)
    
    double A = 0.1;  // Code-specific constant
    double exponent = (distance + 1.0) / 2.0;
    
    if (code == StabilizerCodeType::REPETITION) {
        exponent = static_cast<double>(distance);
        A = 0.5;  // Higher threshold for repetition
    }
    
    return A * std::pow(p_phys, exponent);
}
```

**Test:** `test_adaptive_code_selector.cpp`
- **Test 1:** Biased noise (T1=100µs, T2=20µs) → selects REPETITION
- **Test 2:** High gate error (p=0.06) → selects distance-7 or CONCATENATED
- **Test 3:** Correlated noise → selects SURFACE
- **Test 4:** Balanced noise → selects SURFACE (default)
- **Test 5:** Verify `predict_logical_error_rate` < p_phys² for d=3

---

### 3.3 ML Decoder Architecture

**Purpose:** Neural network-based syndrome decoder with higher accuracy on correlated noise.

#### 3.3.1 Training Data Generation

**Script:** `scripts/generate_qec_training_data.py`

```python
#!/usr/bin/env python3
"""
Generate training dataset for ML QEC decoder.

Output format:
- syndromes.npy: (N, 2*num_stabilizers) array (X + Z syndromes)
- errors.npy: (N, num_data_qubits) array (error locations, multi-hot)
- metadata.json: code parameters, noise profile
"""

import numpy as np
import json
from typing import Tuple

def generate_training_data(
    code_distance: int,
    num_samples: int,
    noise_profile: dict,
    output_dir: str
) -> None:
    """
    Generate synthetic error patterns and corresponding syndromes.
    
    Args:
        code_distance: QEC code distance (3, 5, 7)
        num_samples: Number of training samples (recommend 1M-10M)
        noise_profile: Noise parameters (from Phase 4 calibration)
        output_dir: Directory to save .npy files
    """
    # Load LRET simulator via Python bindings (Phase 5)
    import qlret
    
    # Create surface code
    code = qlret.SurfaceCode(code_distance)
    
    syndromes = []
    errors = []
    
    for i in range(num_samples):
        # Sample error from noise model
        error = sample_error_pattern(code, noise_profile)
        
        # Extract syndrome (with measurement noise)
        syndrome = extract_syndrome_with_noise(code, error, noise_profile)
        
        syndromes.append(syndrome)
        errors.append(error)
        
        if (i + 1) % 100000 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    # Save to disk
    np.save(f"{output_dir}/syndromes.npy", np.array(syndromes))
    np.save(f"{output_dir}/errors.npy", np.array(errors))
    
    metadata = {
        "code_distance": code_distance,
        "num_samples": num_samples,
        "noise_profile": noise_profile,
        "code_type": "surface"
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def sample_error_pattern(code, noise_profile: dict) -> np.ndarray:
    """
    Sample realistic error pattern from noise model.
    
    Includes:
    - Depolarizing errors (iid)
    - T1/T2 decay
    - Correlated errors (if present)
    - Gate-dependent errors
    """
    n_data = code.num_data_qubits()
    error = np.zeros(n_data, dtype=int)  # 0=I, 1=X, 2=Z, 3=Y
    
    p = noise_profile.get("avg_gate_error", 0.001)
    
    # Independent depolarizing
    for q in range(n_data):
        if np.random.rand() < p:
            error[q] = np.random.randint(1, 4)  # X, Z, or Y
    
    # Add correlated errors (10% of time)
    if "correlated_errors" in noise_profile and np.random.rand() < 0.1:
        add_correlated_error(error, noise_profile["correlated_errors"])
    
    return error

def extract_syndrome_with_noise(
    code, 
    error: np.ndarray, 
    noise_profile: dict
) -> np.ndarray:
    """
    Extract syndrome with measurement errors.
    """
    # Get ideal syndrome
    syndrome_x = code.measure_x_stabilizers(error)
    syndrome_z = code.measure_z_stabilizers(error)
    
    # Add measurement noise
    p_meas = noise_profile.get("avg_readout_error", 0.001)
    
    for i in range(len(syndrome_x)):
        if np.random.rand() < p_meas:
            syndrome_x[i] ^= 1
    
    for i in range(len(syndrome_z)):
        if np.random.rand() < p_meas:
            syndrome_z[i] ^= 1
    
    return np.concatenate([syndrome_x, syndrome_z])
```

**Training Pipeline:**

```bash
# Generate 1M training samples for distance-5 surface code
python scripts/generate_qec_training_data.py \
    --code-distance 5 \
    --num-samples 1000000 \
    --noise-profile calibration_outputs/ibmq_jakarta_noise.json \
    --output-dir training_data/d5_ibmq
```

#### 3.3.2 ML Model Architecture

**Script:** `scripts/train_ml_decoder.py`

```python
#!/usr/bin/env python3
"""
Train ML decoder for QEC syndrome decoding.

Supports JAX (Flax) and PyTorch backends.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Tuple
import numpy as np

class TransformerDecoder(nn.Module):
    """
    Transformer-based QEC decoder.
    
    Architecture:
    - Syndrome embedding (linear projection)
    - N transformer blocks (self-attention + FFN)
    - Output head (multi-label classification)
    """
    num_data_qubits: int
    num_stabilizers: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    
    def setup(self):
        # Embedding layer: syndrome bits → hidden_dim
        self.embed = nn.Dense(self.hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(self.hidden_dim, self.num_heads)
            for _ in range(self.num_layers)
        ]
        
        # Output: hidden_dim → num_data_qubits (error locations)
        self.output_head = nn.Dense(self.num_data_qubits * 4)  # 4 Pauli outcomes
    
    def __call__(self, syndrome: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            syndrome: (batch_size, 2*num_stabilizers) binary syndrome
        Returns:
            logits: (batch_size, num_data_qubits, 4) Pauli probabilities
        """
        # Embed syndrome
        x = self.embed(syndrome)  # (B, hidden_dim)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, train=train)
        
        # Output logits
        logits = self.output_head(x)  # (B, num_data_qubits * 4)
        logits = logits.reshape(-1, self.num_data_qubits, 4)
        
        return logits

class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    
    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.mlp = nn.Sequential([
            nn.Dense(self.hidden_dim * 4),
            nn.relu,
            nn.Dense(self.hidden_dim)
        ])
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
    
    def __call__(self, x, train: bool = True):
        # Self-attention with residual
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x

def train_decoder(
    train_data_dir: str,
    val_split: float = 0.1,
    batch_size: int = 256,
    num_epochs: int = 50,
    learning_rate: float = 1e-3
) -> None:
    """
    Train ML decoder on generated data.
    """
    # Load data
    syndromes = np.load(f"{train_data_dir}/syndromes.npy")
    errors = np.load(f"{train_data_dir}/errors.npy")
    
    # Train/val split
    n_val = int(len(syndromes) * val_split)
    train_syndromes, val_syndromes = syndromes[:-n_val], syndromes[-n_val:]
    train_errors, val_errors = errors[:-n_val], errors[-n_val:]
    
    # Initialize model
    with open(f"{train_data_dir}/metadata.json") as f:
        metadata = json.load(f)
    
    code_distance = metadata["code_distance"]
    num_data_qubits = code_distance * code_distance
    num_stabilizers = 2 * (code_distance * code_distance - 1)
    
    model = TransformerDecoder(
        num_data_qubits=num_data_qubits,
        num_stabilizers=num_stabilizers
    )
    
    # Training loop
    optimizer = optax.adam(learning_rate)
    
    # ... standard JAX training loop ...
    
    # Save checkpoint
    save_checkpoint(model, f"checkpoints/decoder_d{code_distance}.pkl")
```

**Model Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 256 | Balance capacity vs speed |
| `num_heads` | 8 | Standard transformer config |
| `num_layers` | 4 | Deep enough for d=3-7 codes |
| `batch_size` | 256 | Fits in GPU memory |
| `learning_rate` | 1e-3 | Adam default |
| `num_epochs` | 50 | Converges in 30-40 typically |

**Expected Performance:**
- Training time: ~2 hours on single GPU (1M samples, d=5)
- Validation accuracy: 92-95% (vs 88-90% MWPM on correlated noise)
- Inference latency: ~2ms per syndrome (batch=1, GPU)

#### 3.3.3 C++ Inference Wrapper

**Header:** `include/qec_adaptive.h`

```cpp
namespace qlret {

class MLDecoder {
public:
    struct Config {
        std::string model_path;              // Path to saved .pkl checkpoint
        std::string backend = "jax";         // "jax" or "pytorch"
        bool use_gpu = true;                 // GPU inference
        size_t batch_size = 1;               // Inference batch size
    };
    
    explicit MLDecoder(const StabilizerCode& code, Config config);
    
    // Main decode interface (compatible with Decoder base class)
    Correction decode(const Syndrome& syndrome) override;
    
    // Batch decode for efficiency
    std::vector<Correction> decode_batch(const std::vector<Syndrome>& syndromes);
    
    // Model management
    void load_model(const std::string& path);
    void reload_model();  // For closed-loop retraining
    
    // Statistics
    struct Stats {
        size_t num_inferences = 0;
        double total_inference_time_ms = 0.0;
        double avg_inference_time_ms() const {
            return num_inferences ? total_inference_time_ms / num_inferences : 0.0;
        }
    };
    Stats stats() const { return stats_; }
    
private:
    const StabilizerCode& code_;
    Config config_;
    Stats stats_;
    
    // Python bridge via pybind11
    std::unique_ptr<PythonMLModel> py_model_;
    
    // Convert syndrome to model input format
    std::vector<float> syndrome_to_tensor(const Syndrome& syndrome);
    
    // Convert model output to Correction
    Correction tensor_to_correction(const std::vector<float>& logits);
};

} // namespace qlret
```

**Implementation:** `src/qec_adaptive.cpp`

```cpp
#include "qec_adaptive.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

MLDecoder::MLDecoder(const StabilizerCode& code, Config config)
    : code_(code), config_(config) {
    load_model(config.model_path);
}

void MLDecoder::load_model(const std::string& path) {
    // Initialize Python interpreter (if not already)
    static py::scoped_interpreter guard{};
    
    try {
        py::module_ pickle = py::module_::import("pickle");
        py::module_ jax = py::module_::import("jax");
        
        // Load model from pickle
        py::object model_file = py::open(path, "rb");
        py_model_ = pickle.attr("load")(model_file).cast<PythonMLModel*>();
        
    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Failed to load ML model: " + std::string(e.what()));
    }
}

Correction MLDecoder::decode(const Syndrome& syndrome) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert syndrome to tensor
    std::vector<float> input = syndrome_to_tensor(syndrome);
    
    // Call Python model
    py::array_t<float> input_array({1, input.size()}, input.data());
    py::array_t<float> output_array = py_model_->attr("predict")(input_array);
    
    // Convert to correction
    auto output_ptr = output_array.data();
    std::vector<float> logits(output_ptr, output_ptr + output_array.size());
    Correction correction = tensor_to_correction(logits);
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats_.num_inferences++;
    stats_.total_inference_time_ms += time_ms;
    
    return correction;
}

std::vector<float> MLDecoder::syndrome_to_tensor(const Syndrome& syndrome) {
    // Concatenate X and Z syndromes
    std::vector<float> tensor;
    tensor.reserve(syndrome.x_syndrome.size() + syndrome.z_syndrome.size());
    
    for (int s : syndrome.x_syndrome) {
        tensor.push_back(static_cast<float>(s));
    }
    for (int s : syndrome.z_syndrome) {
        tensor.push_back(static_cast<float>(s));
    }
    
    return tensor;
}

Correction MLDecoder::tensor_to_correction(const std::vector<float>& logits) {
    // logits: (num_data_qubits, 4) flattened
    // Take argmax for each qubit
    
    size_t n_qubits = code_.num_data_qubits();
    Correction correction;
    correction.x_correction = PauliString(n_qubits);
    correction.z_correction = PauliString(n_qubits);
    
    for (size_t q = 0; q < n_qubits; ++q) {
        // Get logits for this qubit (4 Pauli outcomes)
        size_t offset = q * 4;
        int max_idx = 0;
        float max_val = logits[offset];
        
        for (int i = 1; i < 4; ++i) {
            if (logits[offset + i] > max_val) {
                max_val = logits[offset + i];
                max_idx = i;
            }
        }
        
        // Map index to Pauli: 0=I, 1=X, 2=Z, 3=Y
        if (max_idx == 1 || max_idx == 3) {
            correction.x_correction.set(q, Pauli::X);
        }
        if (max_idx == 2 || max_idx == 3) {
            correction.z_correction.set(q, Pauli::Z);
        }
    }
    
    return correction;
}
```

**Test:** `test_ml_decoder.cpp`
- Load pre-trained model from `test_models/decoder_d3.pkl`
- Decode 100 test syndromes
- Compare with MWPM decoder
- Verify accuracy ≥ 90%
- Measure inference latency < 5ms

---

### 3.4 Closed-Loop Calibration Controller

**Purpose:** Monitor QEC performance and trigger recalibration when drift detected.

**Header:** `include/qec_adaptive.h`

```cpp
namespace qlret {

class ClosedLoopController {
public:
    struct Config {
        size_t window_size = 100;                 // Moving average window
        double drift_threshold = 0.15;            // Relative change threshold (15%)
        size_t recalibration_interval = 1000;     // Check every N cycles
        bool auto_recalibrate = true;             // Trigger Phase 4 automatically
        std::string calibration_script_path = "scripts/calibrate_noise_model.py";
    };
    
    explicit ClosedLoopController(Config config = {});
    
    // Update with QEC round result
    void update(const QECRoundResult& result);
    
    // Check if recalibration needed
    bool should_recalibrate() const;
    
    // Trigger recalibration (calls Phase 4.2 script)
    NoiseProfile recalibrate();
    
    // Update decoder with new noise profile
    void update_decoder(MLDecoder& decoder, const NoiseProfile& new_noise);
    
    // Statistics
    struct Stats {
        size_t total_cycles = 0;
        size_t logical_errors = 0;
        double current_logical_error_rate = 0.0;
        double baseline_logical_error_rate = 0.0;
        size_t num_recalibrations = 0;
        double avg_logical_error_rate() const {
            return total_cycles ? static_cast<double>(logical_errors) / total_cycles : 0.0;
        }
    };
    Stats stats() const { return stats_; }
    
private:
    Config config_;
    Stats stats_;
    
    // Moving average tracking
    std::deque<bool> recent_errors_;  // Circular buffer
    
    // Drift detection
    bool detect_drift() const;
    double compute_current_rate() const;
};

} // namespace qlret
```

**Implementation:** `src/qec_adaptive.cpp`

```cpp
ClosedLoopController::ClosedLoopController(Config config)
    : config_(config) {
    // Initialize baseline from first window
}

void ClosedLoopController::update(const QECRoundResult& result) {
    stats_.total_cycles++;
    
    if (result.logical_error) {
        stats_.logical_errors++;
        recent_errors_.push_back(true);
    } else {
        recent_errors_.push_back(false);
    }
    
    // Maintain window size
    if (recent_errors_.size() > config_.window_size) {
        bool removed = recent_errors_.front();
        recent_errors_.pop_front();
        if (removed) {
            // Adjust if needed for exact tracking
        }
    }
    
    // Update current rate
    stats_.current_logical_error_rate = compute_current_rate();
    
    // Set baseline after first window
    if (stats_.total_cycles == config_.window_size) {
        stats_.baseline_logical_error_rate = stats_.current_logical_error_rate;
    }
}

bool ClosedLoopController::should_recalibrate() const {
    // Check interval
    if (stats_.total_cycles % config_.recalibration_interval != 0) {
        return false;
    }
    
    // Check if enough data
    if (recent_errors_.size() < config_.window_size) {
        return false;
    }
    
    return detect_drift();
}

bool ClosedLoopController::detect_drift() const {
    // Compare current rate to baseline
    double current = stats_.current_logical_error_rate;
    double baseline = stats_.baseline_logical_error_rate;
    
    if (baseline < 1e-9) {
        return false;  // Baseline not established
    }
    
    double relative_change = std::abs(current - baseline) / baseline;
    return relative_change > config_.drift_threshold;
}

double ClosedLoopController::compute_current_rate() const {
    if (recent_errors_.empty()) return 0.0;
    
    size_t errors = 0;
    for (bool e : recent_errors_) {
        if (e) errors++;
    }
    
    return static_cast<double>(errors) / recent_errors_.size();
}

NoiseProfile ClosedLoopController::recalibrate() {
    // Call Phase 4.2 calibration script
    std::string cmd = "python " + config_.calibration_script_path + 
                      " --output recalibration_output.json";
    
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("Calibration script failed");
    }
    
    // Load new noise profile
    NoiseProfile new_profile = NoiseProfile::load_from_calibration(
        "recalibration_output.json"
    );
    
    stats_.num_recalibrations++;
    
    // Update baseline
    stats_.baseline_logical_error_rate = stats_.current_logical_error_rate;
    
    return new_profile;
}

void ClosedLoopController::update_decoder(
    MLDecoder& decoder, 
    const NoiseProfile& new_noise
) {
    // Option 1: Retrain decoder (slow, ~2 hours)
    // Option 2: Fine-tune existing decoder (fast, ~10 min)
    // Option 3: Load pre-trained decoder for similar noise profile
    
    // For now: assume pre-trained models exist for common noise profiles
    std::string model_path = select_pretrained_model(new_noise);
    decoder.load_model(model_path);
}
```

**Test:** `test_closed_loop_controller.cpp`
- Simulate 1000 QEC cycles with constant error rate
- Inject drift at cycle 500 (increase error rate by 20%)
- Verify `should_recalibrate()` triggers after window fills
- Mock recalibration (return fixed NoiseProfile)
- Verify new baseline updated correctly

---

## 4. Integration Points

### 4.1 Phase 4 Integration (Noise Calibration)

**Data Flow:**
```
Phase 4.2 (calibrate_noise_model.py) 
  → outputs/ibmq_device_noise.json
  → NoiseProfile::load_from_calibration()
  → AdaptiveCodeSelector::select_code()
```

**Required Changes to Phase 4:**
1. Standardize JSON output format from `calibrate_noise_model.py`
2. Include all fields needed by NoiseProfile (T1, T2, gate errors, correlations)
3. Add `--output-format adaptive` flag to generate QEC-ready JSON

**Example JSON Format:**

```json
{
  "calibration_timestamp": "2026-01-06T10:30:00Z",
  "device_name": "ibmq_jakarta",
  "num_qubits": 7,
  "t1_times_ns": [55000, 62000, 48000, ...],
  "t2_times_ns": [75000, 80000, 65000, ...],
  "single_gate_errors": [0.0012, 0.0015, 0.0018, ...],
  "two_qubit_errors": {
    "(0, 1)": 0.009,
    "(1, 2)": 0.011,
    ...
  },
  "readout_errors": [0.015, 0.018, 0.020, ...],
  "correlated_errors": [],
  "time_varying": null
}
```

### 4.2 Phase 5 Integration (Python Bindings)

**Required Bindings:**

```cpp
// python/bindings.cpp
PYBIND11_MODULE(_qlret_native, m) {
    // ... existing bindings ...
    
    // Phase 9.3: Adaptive QEC
    py::class_<NoiseProfile>(m, "NoiseProfile")
        .def(py::init<>())
        .def("load_from_calibration", &NoiseProfile::load_from_calibration)
        .def("to_json", &NoiseProfile::to_json)
        .def("avg_gate_error", &NoiseProfile::avg_gate_error)
        .def("is_biased", &NoiseProfile::is_biased);
    
    py::class_<AdaptiveCodeSelector>(m, "AdaptiveCodeSelector")
        .def(py::init<>())
        .def("select_code", &AdaptiveCodeSelector::select_code)
        .def("predict_logical_error_rate", &AdaptiveCodeSelector::predict_logical_error_rate)
        .def("select_distance", &AdaptiveCodeSelector::select_distance);
    
    py::class_<MLDecoder, Decoder>(m, "MLDecoder")
        .def(py::init<const StabilizerCode&, MLDecoder::Config>())
        .def("decode", &MLDecoder::decode)
        .def("load_model", &MLDecoder::load_model);
    
    py::class_<ClosedLoopController>(m, "ClosedLoopController")
        .def(py::init<>())
        .def("update", &ClosedLoopController::update)
        .def("should_recalibrate", &ClosedLoopController::should_recalibrate)
        .def("recalibrate", &ClosedLoopController::recalibrate)
        .def("stats", &ClosedLoopController::stats);
}
```

**Python Usage Example:**

```python
import qlret

# Load noise profile from Phase 4 calibration
noise = qlret.NoiseProfile.load_from_calibration("calibration_output.json")

# Select optimal code
selector = qlret.AdaptiveCodeSelector()
code_type = selector.select_code(noise)
distance = selector.select_distance(code_type, noise, target_error_rate=1e-6)

print(f"Selected: {code_type} with distance {distance}")

# Create QEC system with ML decoder
config = qlret.MLDecoder.Config()
config.model_path = "models/decoder_d5.pkl"
decoder = qlret.MLDecoder(code, config)

# Run closed-loop QEC
controller = qlret.ClosedLoopController()
for cycle in range(10000):
    result = logical_qubit.qec_round()
    controller.update(result)
    
    if controller.should_recalibrate():
        new_noise = controller.recalibrate()
        new_code_type = selector.select_code(new_noise)
        # Recreate code if needed
```

### 4.3 Phase 8 Integration (Fault Tolerance)

**Checkpoint Extension:**

Add adaptive QEC state to checkpoint format:

```cpp
struct QECCheckpoint {
    // Existing fields from Phase 9.1/9.2
    size_t cycle_number;
    std::vector<Syndrome> syndrome_history;
    PauliString accumulated_error;
    
    // Phase 9.3: Adaptive state
    StabilizerCodeType active_code_type;
    size_t active_distance;
    std::string active_decoder_model_path;
    NoiseProfile current_noise_profile;
    
    // Closed-loop stats
    ClosedLoopController::Stats controller_stats;
};
```

**Recovery Logic:**

```cpp
bool FaultTolerantQECRunner::recover(const std::string& path) {
    QECCheckpoint ckpt = load_checkpoint(path);
    
    // Restore code and decoder
    logical_qubit_ = create_logical_qubit(ckpt.active_code_type, ckpt.active_distance);
    
    if (use_ml_decoder_) {
        ml_decoder_ = std::make_unique<MLDecoder>(
            logical_qubit_->code(),
            MLDecoder::Config{ckpt.active_decoder_model_path}
        );
    }
    
    // Restore controller state
    closed_loop_controller_.restore_stats(ckpt.controller_stats);
    
    return true;
}
```

---

## 5. Testing Strategy

### 5.1 Unit Tests (52 tests)

**File:** `tests/test_qec_adaptive.cpp`

| Test Name | Purpose | Success Criteria |
|-----------|---------|------------------|
| `test_noise_profile_load` | Load JSON from Phase 4 | All fields parsed correctly |
| `test_noise_profile_statistics` | Compute avg/max errors | Matches manual calculation |
| `test_noise_profile_bias_detection` | Detect T1/T2 bias | Correct for T1=2*T2 |
| `test_selector_biased_noise` | Select code for bias | REPETITION chosen |
| `test_selector_high_error` | Select for high errors | Distance-7 or CONCATENATED |
| `test_selector_correlated_noise` | Select for correlations | SURFACE chosen |
| `test_selector_default` | Balanced noise | SURFACE chosen |
| `test_selector_predict_logical_error` | Error rate prediction | p_L < p_phys² for d=3 |
| `test_selector_distance_selection` | Choose distance | Meets target error rate |
| `test_ml_decoder_load_model` | Load .pkl checkpoint | No exceptions |
| `test_ml_decoder_inference` | Decode syndrome | Produces valid correction |
| `test_ml_decoder_batch` | Batch inference | Faster than serial |
| `test_ml_decoder_accuracy` | Compare to MWPM | Accuracy ≥ 90% |
| `test_closed_loop_update` | Update stats | Correct rate computation |
| `test_closed_loop_drift_detection` | Detect 20% drift | Triggers recalibration |
| `test_closed_loop_recalibrate` | Mock recalibration | New noise profile loaded |
| `test_closed_loop_baseline_update` | Update baseline | Baseline = current after recalib |

### 5.2 Integration Tests (15 tests)

**File:** `tests/test_qec_adaptive_integration.cpp`

| Test Name | Purpose | Success Criteria |
|-----------|---------|------------------|
| `test_adaptive_qec_pipeline` | Full adaptive QEC cycle | Completes without crash |
| `test_code_switching` | Switch codes mid-simulation | New code loaded correctly |
| `test_ml_decoder_vs_mwpm` | Compare decoders | ML improves fidelity by ≥5% |
| `test_closed_loop_1000_cycles` | Long-running closed loop | Drift detected and corrected |
| `test_phase4_integration` | Load real calibration data | NoiseProfile valid |
| `test_checkpoint_adaptive_state` | Save/load adaptive state | Recovery restores code/decoder |
| `test_distributed_adaptive_qec` | Multi-rank with adaptive | All ranks select same code |

### 5.3 End-to-End Tests (5 tests)

**File:** `tests/test_qec_e2e.cpp`

| Test Name | Purpose | Success Criteria |
|-----------|---------|------------------|
| `test_10k_cycle_surface_code` | Sustained execution | p_L < 0.01 at p_phys=0.001 |
| `test_noise_drift_handling` | Inject drift at cycle 5k | Recalibration stabilizes p_L |
| `test_ml_decoder_training_pipeline` | Generate data → train → deploy | Accuracy ≥ 92% on test set |

---

## 6. Performance Targets

### 6.1 Code Selection

| Metric | Target | Measurement |
|--------|--------|-------------|
| Selection latency | < 1ms | Time from NoiseProfile → code type |
| Prediction accuracy | ≥80% | % correct code choice vs oracle |

### 6.2 ML Decoder

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training time (d=5, 1M samples) | < 3 hours | Single GPU |
| Inference latency | < 5ms | Single syndrome, GPU |
| Batch inference latency | < 2ms/syndrome | Batch size = 32 |
| Accuracy (iid noise) | ≥ 90% | Match or beat MWPM |
| Accuracy (correlated noise) | ≥ 95% | 5% improvement over MWPM |

### 6.3 Closed-Loop Controller

| Metric | Target | Measurement |
|--------|--------|-------------|
| Drift detection latency | < 100 cycles | Window size = 100 |
| Recalibration time | < 5 minutes | Phase 4 script + model reload |
| False positive rate | < 5% | Triggers without real drift |
| False negative rate | < 1% | Misses 20% drift |

---

## 7. Implementation Timeline (2-3 weeks)

### Week 1: Foundation (Days 1-7)

**Day 1-2: NoiseProfile & AdaptiveCodeSelector**
- Implement NoiseProfile data structure
- JSON serialization/deserialization
- AdaptiveCodeSelector with decision tree
- Unit tests (10 tests)

**Day 3-4: Training Data Generation**
- `generate_qec_training_data.py` script
- Error sampling from noise model
- Syndrome extraction with measurement noise
- Generate 100k samples for d=3 (test dataset)

**Day 5-7: ML Decoder (Python)**
- TransformerDecoder architecture (JAX/Flax)
- Training loop with validation
- Save/load checkpoints
- Train initial model for d=3

### Week 2: Integration (Days 8-14)

**Day 8-9: ML Decoder (C++ Wrapper)**
- MLDecoder C++ class
- pybind11 bridge
- Inference pipeline
- Unit tests (8 tests)

**Day 10-11: Closed-Loop Controller**
- ClosedLoopController implementation
- Drift detection algorithm
- Mock recalibration for testing
- Unit tests (7 tests)

**Day 12-14: Phase 4/5/8 Integration**
- Update Phase 4 JSON output format
- Add Python bindings
- Extend checkpoint format
- Integration tests (15 tests)

### Week 3: Validation (Days 15-21)

**Day 15-17: End-to-End Testing**
- 10k cycle simulation
- Noise drift handling
- Multi-decoder comparison
- E2E tests (5 tests)

**Day 18-19: Performance Optimization**
- Batch inference for ML decoder
- Code selection caching
- Profiling and bottleneck analysis

**Day 20-21: Documentation**
- Update TESTING_BACKLOG.md
- API documentation
- User guide for adaptive QEC
- Performance benchmarks

---

## 8. Deliverables

### 8.1 Source Code

- `include/qec_adaptive.h` (~400 lines)
- `src/qec_adaptive.cpp` (~800 lines)
- `scripts/generate_qec_training_data.py` (~300 lines)
- `scripts/train_ml_decoder.py` (~400 lines)
- `python/qlret/qec_adaptive.py` (~200 lines)

### 8.2 Tests

- `tests/test_qec_adaptive.cpp` (~700 lines, 52 tests)
- `tests/test_qec_adaptive_integration.cpp` (~500 lines, 15 tests)
- `tests/test_qec_e2e.cpp` (~300 lines, 5 tests)

### 8.3 Models & Data

- `models/decoder_d3.pkl` (pre-trained model for distance-3)
- `models/decoder_d5.pkl` (pre-trained model for distance-5)
- `training_data/d3_synthetic/` (100k samples for testing)

### 8.4 Documentation

- `docs/adaptive-qec-guide.md` (user guide)
- `docs/ml-decoder-training.md` (training pipeline)
- Update `TESTING_BACKLOG.md` with Phase 9.3 tests

---

## 9. Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| ML decoder underfits | High | Low | Use proven transformer architecture, sufficient training data |
| pybind11 memory leaks | Medium | Medium | Careful lifetime management, use smart pointers |
| Recalibration too slow | Medium | Medium | Use incremental updates, pre-trained model library |
| Code switching breaks mid-sim | High | Low | Extensive checkpoint/recovery testing |
| Training data not diverse | High | Medium | Include correlated errors, measurement noise, time-varying effects |

---

## 10. Success Criteria (Phase 9.3 Complete)

- ✅ **AdaptiveCodeSelector** selects correct code for 80% of noise profiles
- ✅ **ML decoder** achieves ≥95% accuracy on correlated noise (vs 90% MWPM)
- ✅ **Closed-loop** maintains p_L within 10% of baseline under 20% noise drift
- ✅ **Integration** with Phase 4, 5, 8 seamless (all tests pass)
- ✅ **E2E test:** 10k QEC cycles complete with adaptive code selection and ML decoding
- ✅ **Performance:** Inference < 5ms, selection < 1ms, recalibration < 5 min

---

## 11. Post-Phase 9.3: Path to Phase 10

**Next Steps:**
1. Large-scale benchmarking (d=7-9 codes, 100k+ cycles)
2. Multi-code comparison (Surface vs Repetition vs Color on real devices)
3. Publish ML decoder training pipeline as standalone tool
4. Integrate with PennyLane (adaptive QEC for VQE/QAOA)

**Ready to proceed with implementation!** 🚀
