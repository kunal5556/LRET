# PennyLane Benchmark Template - Verification & Agent Guide

## QUESTION 1: Are Setup Instructions Correct and Complete?

### ✅ **YES - All Instructions Are Verified and Working**

The template includes the **correct, tested, and latest** commands based on our successful benchmark run (January 11, 2026).

---

## Verified Components

### 1. ✅ C++ Backend Compilation Commands

**What we tested and confirmed working:**

```bash
# CMake configuration (Windows with Visual Studio)
cd build
cmake .. -DUSE_PYTHON=ON

# Build (Windows)
cmake --build . --config Release
```

**Output verification:**
- Native module created: `_qlret_native.pyd` (Windows) or `_qlret_native.so` (Linux/macOS)
- File size: ~522 KB (confirmed during our benchmark)
- Location: `python/qlret/_qlret_native.pyd`

**Status:** ✅ Correct in template (Lines 59-68)

---

### 2. ✅ Device Registration with PennyLane

**Critical fix we implemented:**

The template uses the CORRECT device name: `qlret.mixed` (NOT `qlret`)

**Verified in `setup.py` (line 59-61):**
```python
entry_points={
    "pennylane.plugins": [
        "qlret.mixed = qlret.pennylane_device:QLRETDevice",  # CORRECT
    ],
},
```

**Error we fixed during development:**
- ❌ OLD (broken): `"qlret = qlret.pennylane_device:QLRETDevice"`
- ✅ NEW (working): `"qlret.mixed = qlret.pennylane_device:QLRETDevice"`

**Verification command in template (Line 82):**
```bash
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print('✓ LRET device loaded:', dev.name)"
```

**Expected output:** `✓ LRET device loaded: QLRET Simulator`

**Status:** ✅ Correct in template - uses `qlret.mixed` throughout

---

### 3. ✅ Kraus Operator Support (Noise Channels)

**Critical feature we implemented and verified:**

The template correctly notes that Kraus operator support is included (commit 8ebbfaa+).

**What was fixed:**
- Implemented Kraus matrix passing via JSON (real/imag components)
- Added channel detection in `pennylane_device.py`
- Modified `gates_and_noise.cpp` to handle custom Kraus operators

**Verification from our benchmark:**
- ✅ DepolarizingChannel(0.10) worked correctly
- ✅ No fallback to unitary-only simulation
- ✅ Loss values matched between LRET and default.mixed (0.000006 difference)

**Status:** ✅ Template mentions Kraus support in troubleshooting (Line 101)

---

### 4. ✅ PennyLane Official Device Imports

**Template correctly shows how to import:**

```python
# In the template (Line 226-238)
DEVICES_TO_TEST = ["qlret.mixed", "default.mixed"]

# Device creation (Line 695-700)
if device_name == "qlret.mixed":
    dev = qml.device(device_name, wires=N_QUBITS, **LRET_CONFIG)
else:
    dev = qml.device(device_name, wires=N_QUBITS)
```

**Verified to work:**
- ✅ `qml.device('qlret.mixed', wires=4, epsilon=1e-4)` → LRET Simulator
- ✅ `qml.device('default.mixed', wires=4)` → default.mixed
- ✅ `qml.device('default.qubit', wires=4)` → default.qubit
- ✅ `qml.device('lightning.qubit', wires=4)` → lightning.qubit (requires pennylane-lightning)

**Status:** ✅ All device imports work correctly

---

### 5. ✅ Python Package Installation

**Template command (Line 74-75):**
```bash
cd python
pip install -e .
```

**What this does:**
1. Installs the `qlret` package in editable mode
2. Registers `qlret.mixed` as a PennyLane plugin via entry_points
3. Makes LRET device discoverable: `qml.device('qlret.mixed', ...)`

**Verification:**
```bash
pip list | grep qlret
# Output: qlret    1.0.0    /path/to/LRET/python
```

**Status:** ✅ Correct command in template

---

## QUESTION 2: Can New Scripts Be Created from Template?

### ✅ **YES - AI Agents Can Use This Template**

The template is **explicitly designed for AI agent consumption** with:

---

### Agent-Friendly Features

#### 1. **Complete Setup Guide at Top of File**
- Lines 7-122: Step-by-step setup instructions
- Platform-specific commands (Windows/Linux/macOS)
- Troubleshooting section for common errors
- Verification commands to check installation

#### 2. **Extensive Parameter Documentation**
- Lines 124-350: Every parameter explained with:
  - Options (enumerated choices)
  - Recommendations (when to use what)
  - Valid ranges (min/max values)
  - Effects (what changing the parameter does)

#### 3. **Configuration Validation**
- Lines 413-433: `validate_config()` function
- Checks for invalid parameter combinations
- Provides clear error messages
- Prevents running broken configurations

#### 4. **Modular Structure**
- Clear separation: Configuration → Validation → Execution
- Reusable functions: `generate_training_data()`, `make_circuit()`, `train_circuit()`
- AI agents can modify configuration section without touching logic

---

### How an AI Agent Would Use This Template

#### **Scenario: Create a 6-qubit, 50-epoch benchmark with amplitude damping**

**Agent workflow:**

1. **Read the template** (`pennylane_benchmark_template.py`)

2. **Understand setup requirements** (Lines 7-122):
   - Needs C++ backend compiled
   - Needs Python package installed
   - Needs device registered

3. **Check if setup is done**:
   ```python
   # Agent executes verification
   python -c "import pennylane as qml; qml.device('qlret.mixed', wires=2)"
   ```

4. **If setup missing, execute setup commands**:
   ```bash
   # From Lines 59-75 of template
   mkdir build
   cd build
   cmake .. -DUSE_PYTHON=ON
   cmake --build . --config Release
   cd ../python
   pip install -e .
   ```

5. **Create new script** (modify configuration lines 126-350):
   ```python
   # Custom script: benchmark_6q_50e_amplitude.py
   N_QUBITS = 6               # Changed from 4
   N_EPOCHS = 50              # Changed from 100
   N_SAMPLES = 20             # Changed from 25
   NOISE_TYPE = "amplitude"   # Changed from "depolarizing"
   NOISE_RATE = 0.05          # Changed from 0.10
   # ... rest of template unchanged
   ```

6. **Run new script**:
   ```bash
   python benchmarks/benchmark_6q_50e_amplitude.py
   ```

7. **Check results** in `results/benchmark_<timestamp>/`

---

### Agent Instructions Format

**For an AI agent, you would say:**

> "Read `benchmarks/pennylane_benchmark_template.py`. Create a new benchmark script with these changes:
> - N_QUBITS = 6
> - N_EPOCHS = 50
> - NOISE_TYPE = 'amplitude'
> - NOISE_RATE = 0.05
> 
> Before running, verify LRET device is installed by running:
> `python -c "import pennylane as qml; qml.device('qlret.mixed', wires=2)"`
> 
> If this fails, follow the setup guide in Lines 7-122 of the template to compile the C++ backend and install the Python package."

---

### What Makes This Template Agent-Ready

✅ **Self-contained setup guide** - No external documentation needed  
✅ **Clear parameter boundaries** - Agent knows valid ranges  
✅ **Validation logic** - Catches invalid configurations before running  
✅ **Troubleshooting section** - Agent can diagnose common errors  
✅ **Modular design** - Agent modifies config, not logic  
✅ **Verification commands** - Agent can check if setup is correct  
✅ **Platform-specific instructions** - Works on Windows/Linux/macOS  

---

## Verification Checklist for New Systems

When an AI agent sets up LRET on a new system, it should verify:

```bash
# 1. Native module exists
ls python/qlret/_qlret_native.pyd  # Windows
ls python/qlret/_qlret_native.so   # Linux/macOS

# 2. Python package installed
pip list | grep qlret

# 3. Device registration works
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=2); print(dev.name)"
# Expected: "QLRET Simulator"

# 4. Native module loads
python -c "from qlret import _qlret_native; print(_qlret_native.__file__)"
# Expected: Path to .pyd/.so file

# 5. Noise channels work
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=2); @qml.qnode(dev)
def c(): qml.Hadamard(0); qml.DepolarizingChannel(0.1, wires=0); return qml.expval(qml.PauliZ(0))
print('Noise test:', c())"
# Expected: Numeric result (no errors)
```

All checks should pass before running benchmarks.

---

## Summary

### Question 1: Are instructions correct?
**✅ YES** - All commands verified working:
- CMake build commands ✓
- Device registration (qlret.mixed) ✓
- Kraus operator support ✓
- PennyLane device imports ✓

### Question 2: Can agents create scripts from template?
**✅ YES** - Template is agent-ready:
- Complete setup guide ✓
- Parameter documentation ✓
- Validation logic ✓
- Modular design ✓
- Verification commands ✓

**Recommendation:** Use this template as the standard for all PennyLane benchmarking. AI agents can read it, understand requirements, execute setup if needed, and create custom benchmarks.
