# QLRET PennyLane Plugin - Bug Fix Summary

## Overview
This document summarizes the comprehensive fixes made to the QLRET PennyLane device plugin to work with PennyLane 0.43+.

## Issues Fixed

### 1. PennyLane 0.43+ API Compatibility

**Problem**: PennyLane 0.43 moved `Device` class from `pennylane` to `pennylane.devices`.

**Fix**: Added try/except import handling in `pennylane_device.py`:
```python
try:
    from pennylane.devices import Device, DeviceCapabilities
    _HAS_DEVICE_CAPABILITIES = True
except ImportError:
    from pennylane import Device
    DeviceCapabilities = None
    _HAS_DEVICE_CAPABILITIES = False
```

### 2. Missing Backend Fallback

**Problem**: When native C++ backend or CLI executable is unavailable, the device threw an error and couldn't run.

**Fix**: Created `fallback_simulator.py` - a pure-Python density matrix simulator that provides:
- Basic quantum gate support (H, X, Y, Z, RX, RY, RZ, CNOT, etc.)
- Expectation value computation
- Automatic fallback when native backend is unavailable

### 3. DeviceCapabilities for PennyLane 0.43+

**Problem**: PennyLane 0.43+ requires devices to declare supported operations via a TOML config file and `capabilities` property.

**Fix**: 
- Created `device_config.toml` with supported gates, observables, and measurements
- Added `config_filepath` class attribute pointing to the TOML file
- Added `capabilities` property returning proper `DeviceCapabilities` object

### 4. Shots Object Handling

**Problem**: PennyLane 0.43+ uses `Shots` object instead of raw integers.

**Fix**: Updated `_tape_to_json()` to handle `Shots` objects:
```python
if hasattr(tape_shots, 'total_shots'):
    shots_val = tape_shots.total_shots
else:
    shots_val = int(tape_shots) if tape_shots else None
```

### 5. Execute Method Signature

**Problem**: Old execute method tried to handle legacy API which no longer applies in PennyLane 0.43+.

**Fix**: Simplified execute method to modern API:
```python
def execute(
    self,
    circuits: Union[QuantumTape, List[QuantumTape]],
    execution_config: Any = None,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
```

### 6. Added setup_execution_config Method

**Problem**: PennyLane 0.43+ expects devices to implement `setup_execution_config()`.

**Fix**: Added method that returns a properly configured `ExecutionConfig` object.

## Files Modified

1. **python/qlret/pennylane_device.py** - Main device implementation
2. **python/qlret/api.py** - Added fallback simulator integration
3. **python/qlret/__init__.py** - Updated exports

## Files Created

1. **python/qlret/fallback_simulator.py** - Pure-Python quantum simulator
2. **python/qlret/device_config.toml** - PennyLane device capabilities
3. **python/test_device_fixes.py** - Comprehensive test script

## Test Results

All tests pass:
- ✓ Package imports
- ✓ Fallback simulator execution
- ✓ Device instantiation
- ✓ Simple circuit execution
- ✓ Parametric circuit execution
- ✓ Gradient computation (parameter-shift)
- ✓ Multiple observables

## Known Limitations

1. **Fallback Simulator Performance**: The pure-Python simulator uses full density matrices (2^n × 2^n) instead of the low-rank representation. For production use, build the native C++ backend.

2. **Automatic Gradient Methods**: PennyLane's automatic `qml.grad()` may not work seamlessly with all configurations. Manual parameter-shift gradients work correctly.

3. **Mid-Circuit Measurements**: Not supported (declared in device_config.toml).

## Usage

```python
import pennylane as qml
from qlret import QLRETDevice

# Create device (will use fallback if native backend not available)
dev = QLRETDevice(wires=4, shots=None, epsilon=1e-4)

@qml.qnode(dev)
def circuit(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Forward pass works
result = circuit(0.5)

# For gradients, use manual parameter-shift:
shift = np.pi / 2
grad = (circuit(theta + shift) - circuit(theta - shift)) / 2
```

## Building Native Backend

For production performance, build the C++ backend:
```bash
cd build
cmake ..
make -j
```

This will provide the low-rank simulation capabilities and significantly improved performance.
