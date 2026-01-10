"""Benchmark configuration and device factory.

This module defines the devices to benchmark against and configuration
parameters for the benchmark suite.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import pennylane as qml


@dataclass
class DeviceConfig:
    """Configuration for a PennyLane device."""
    name: str                          # Display name
    device_name: str                   # PennyLane device name (e.g., "default.qubit")
    device_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_qubits: int = 20               # Maximum qubits this device can handle
    supports_mixed: bool = False       # Supports density matrix / noise
    requires_install: Optional[str] = None  # Package to install if missing
    category: str = "simulator"        # "simulator", "hardware", "gpu"


# =============================================================================
# Available Devices for Benchmarking
# =============================================================================

BENCHMARK_DEVICES: List[DeviceConfig] = [
    # QLRET (our device)
    DeviceConfig(
        name="QLRET (Low-Rank)",
        device_name="qlret.mixed",
        device_kwargs={"epsilon": 1e-4},
        max_qubits=24,
        supports_mixed=True,
        category="simulator",
    ),
    
    # PennyLane built-in devices
    DeviceConfig(
        name="default.qubit",
        device_name="default.qubit",
        device_kwargs={},
        max_qubits=20,
        supports_mixed=False,
        category="simulator",
    ),
    DeviceConfig(
        name="default.mixed",
        device_name="default.mixed",
        device_kwargs={},
        max_qubits=12,  # Full density matrix limits scalability
        supports_mixed=True,
        category="simulator",
    ),
    DeviceConfig(
        name="lightning.qubit",
        device_name="lightning.qubit",
        device_kwargs={},
        max_qubits=22,
        supports_mixed=False,
        requires_install="pennylane-lightning",
        category="simulator",
    ),
    
    # Qiskit backends (if available)
    DeviceConfig(
        name="qiskit.aer",
        device_name="qiskit.aer",
        device_kwargs={"method": "statevector"},
        max_qubits=20,
        supports_mixed=False,
        requires_install="pennylane-qiskit",
        category="simulator",
    ),
    DeviceConfig(
        name="qiskit.aer (density_matrix)",
        device_name="qiskit.aer",
        device_kwargs={"method": "density_matrix"},
        max_qubits=12,
        supports_mixed=True,
        requires_install="pennylane-qiskit",
        category="simulator",
    ),
    
    # Cirq (if available)
    DeviceConfig(
        name="cirq.simulator",
        device_name="cirq.simulator",
        device_kwargs={},
        max_qubits=20,
        supports_mixed=False,
        requires_install="pennylane-cirq",
        category="simulator",
    ),
]


def get_available_devices(
    require_mixed: bool = False,
    max_qubits: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> List[DeviceConfig]:
    """Get list of available devices that can be instantiated.
    
    Parameters
    ----------
    require_mixed : bool
        If True, only return devices that support mixed states/noise.
    max_qubits : int, optional
        Only return devices that can handle at least this many qubits.
    categories : list, optional
        Filter by device categories (e.g., ["simulator"]).
    
    Returns
    -------
    List[DeviceConfig]
        Available device configurations.
    """
    available = []
    
    for config in BENCHMARK_DEVICES:
        # Check category filter
        if categories and config.category not in categories:
            continue
            
        # Check mixed state support
        if require_mixed and not config.supports_mixed:
            continue
            
        # Check qubit limit
        if max_qubits and config.max_qubits < max_qubits:
            continue
        
        # Try to instantiate the device
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dev = qml.device(config.device_name, wires=2, **config.device_kwargs)
                del dev
            available.append(config)
        except Exception as e:
            # Device not available
            pass
    
    return available


def create_device(config: DeviceConfig, wires: int, shots: Optional[int] = None) -> Any:
    """Create a PennyLane device from configuration.
    
    Parameters
    ----------
    config : DeviceConfig
        Device configuration.
    wires : int
        Number of qubits.
    shots : int, optional
        Number of measurement shots.
    
    Returns
    -------
    PennyLane device instance.
    """
    kwargs = dict(config.device_kwargs)
    if shots is not None:
        kwargs["shots"] = shots
    
    return qml.device(config.device_name, wires=wires, **kwargs)


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Qubit ranges to test
    qubit_range: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 14])
    
    # Number of trials for statistics
    num_trials: int = 3
    
    # Timeout per circuit execution (seconds)
    timeout_seconds: float = 300.0
    
    # Memory limit (GB) - skip if exceeded
    memory_limit_gb: float = 16.0
    
    # Whether to save intermediate results
    save_intermediate: bool = True
    
    # Output directory
    output_dir: str = "benchmark_results"
    
    # Circuits to benchmark
    circuit_types: List[str] = field(default_factory=lambda: [
        "random_circuit",
        "qft",
        "qaoa",
        "vqe",
        "qnn",
        "grover",
    ])
    
    # Depth multipliers for variational circuits
    depth_multipliers: List[int] = field(default_factory=lambda: [1, 2, 3])


# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()


# Quick test configuration (for local testing)
QUICK_TEST_CONFIG = BenchmarkConfig(
    qubit_range=[4, 6, 8],
    num_trials=1,
    timeout_seconds=60.0,
    circuit_types=["random_circuit", "qnn"],
    depth_multipliers=[1],
)
