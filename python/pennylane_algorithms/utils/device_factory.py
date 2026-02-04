"""
Device Factory for LRET PennyLane Algorithm Testing

Creates and configures quantum devices with various modes:
- LRET device modes: sequential, batched, parallel, openmp
- Comparison devices: default.mixed, default.qubit, lightning.qubit
"""

import pennylane as qml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import sys
import os

# Add parent path for qlret import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class DeviceConfig:
    """Configuration for a quantum device."""
    
    name: str
    wires: int
    mode: str = 'default'
    
    # LRET-specific options
    epsilon: float = 1e-4
    max_rank: Optional[int] = None
    use_openmp: bool = False
    batch_size: int = 10
    n_workers: int = 4
    
    # Noise options
    with_noise: bool = False
    noise_strength: float = 0.01
    noise_model: str = 'depolarizing'  # 'depolarizing', 'amplitude_damping', 'phase_damping'
    
    # Additional kwargs
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_device_kwargs(self) -> Dict[str, Any]:
        """Convert to PennyLane device kwargs."""
        kwargs = {'wires': self.wires}
        
        if 'qlret' in self.name:
            kwargs['epsilon'] = self.epsilon
            if self.max_rank is not None:
                kwargs['max_rank'] = self.max_rank
            if self.use_openmp:
                kwargs['use_openmp'] = True
            if self.mode == 'batched':
                kwargs['batch_size'] = self.batch_size
            if self.mode == 'parallel':
                kwargs['n_workers'] = self.n_workers
        
        kwargs.update(self.extra_kwargs)
        return kwargs


# Available LRET device modes
LRET_MODES = {
    'sequential': {
        'description': 'Sequential execution, single-threaded',
        'kwargs': {}
    },
    'batched': {
        'description': 'Batched execution for parameter sweeps',
        'kwargs': {'batch_size': 10}
    },
    'parallel': {
        'description': 'Parallel execution with Python multiprocessing',
        'kwargs': {'n_workers': 4}
    },
    'openmp': {
        'description': 'OpenMP-parallelized C++ backend',
        'kwargs': {'use_openmp': True}
    },
    'low_rank': {
        'description': 'Aggressive low-rank truncation for memory efficiency',
        'kwargs': {'epsilon': 1e-3}
    },
    'high_precision': {
        'description': 'High precision with minimal truncation',
        'kwargs': {'epsilon': 1e-6}
    },
}


# Available comparison devices
COMPARISON_DEVICES = {
    'default.mixed': {
        'description': 'PennyLane default mixed-state simulator (density matrix)',
        'supports_noise': True,
        'kwargs': {}
    },
    'default.qubit': {
        'description': 'PennyLane default pure-state simulator (statevector)',
        'supports_noise': False,
        'kwargs': {}
    },
    'lightning.qubit': {
        'description': 'High-performance C++ statevector simulator',
        'supports_noise': False,
        'kwargs': {}
    },
}


def create_lret_device(
    wires: int,
    mode: str = 'sequential',
    epsilon: float = 1e-4,
    with_noise: bool = False,
    **kwargs
):
    """
    Create an LRET device with specified mode.
    
    Args:
        wires: Number of qubits
        mode: One of 'sequential', 'batched', 'parallel', 'openmp', 'low_rank', 'high_precision'
        epsilon: Truncation threshold (for LRET)
        with_noise: Whether to enable noise (LRET always supports noise)
        **kwargs: Additional device arguments
    
    Returns:
        PennyLane device instance
    """
    # Get mode-specific kwargs
    mode_config = LRET_MODES.get(mode, LRET_MODES['sequential'])
    device_kwargs = {
        'wires': wires,
        'epsilon': epsilon,
        **mode_config['kwargs'],
        **kwargs
    }
    
    try:
        from qlret import QLRETDevice
        dev = qml.device('qlret.mixed', **device_kwargs)
        dev._lret_mode = mode  # Tag for identification
        return dev
    except Exception as e:
        print(f"Warning: Could not create LRET device: {e}")
        print("Falling back to default.mixed")
        return qml.device('default.mixed', wires=wires)


def create_comparison_device(
    device_name: str,
    wires: int,
    with_noise: bool = False,
    **kwargs
):
    """
    Create a comparison device (non-LRET).
    
    Args:
        device_name: One of 'default.mixed', 'default.qubit', 'lightning.qubit'
        wires: Number of qubits
        with_noise: Whether noise is needed (selects appropriate device)
        **kwargs: Additional device arguments
    
    Returns:
        PennyLane device instance
    """
    config = COMPARISON_DEVICES.get(device_name)
    
    if config is None:
        raise ValueError(f"Unknown device: {device_name}. "
                        f"Available: {list(COMPARISON_DEVICES.keys())}")
    
    # Check noise compatibility
    if with_noise and not config['supports_noise']:
        print(f"Warning: {device_name} doesn't support noise. Disabling noise.")
    
    device_kwargs = {
        'wires': wires,
        **config['kwargs'],
        **kwargs
    }
    
    # Special handling for lightning.qubit
    if device_name == 'lightning.qubit':
        try:
            dev = qml.device('lightning.qubit', **device_kwargs)
        except:
            print("Warning: lightning.qubit not available, using default.qubit")
            dev = qml.device('default.qubit', **device_kwargs)
    else:
        dev = qml.device(device_name, **device_kwargs)
    
    dev._comparison_device = device_name
    return dev


def get_all_lret_modes() -> List[str]:
    """Get list of all available LRET modes."""
    return list(LRET_MODES.keys())


def get_all_comparison_devices() -> List[str]:
    """Get list of all comparison devices."""
    return list(COMPARISON_DEVICES.keys())


def get_device_info(device) -> Dict[str, Any]:
    """Get information about a device."""
    info = {
        'name': device.name,
        'wires': device.num_wires,
        'shots': device.shots,
    }
    
    # Check for LRET mode tag
    if hasattr(device, '_lret_mode'):
        info['mode'] = device._lret_mode
        info['is_lret'] = True
    elif hasattr(device, '_comparison_device'):
        info['mode'] = 'default'
        info['is_lret'] = False
    else:
        info['mode'] = 'unknown'
        info['is_lret'] = False
    
    return info


def create_device_from_config(config: DeviceConfig):
    """Create a device from a DeviceConfig object."""
    if 'qlret' in config.name:
        return create_lret_device(
            wires=config.wires,
            mode=config.mode,
            epsilon=config.epsilon,
            with_noise=config.with_noise,
            **config.extra_kwargs
        )
    else:
        return create_comparison_device(
            device_name=config.name,
            wires=config.wires,
            with_noise=config.with_noise,
            **config.extra_kwargs
        )


def get_recommended_devices_for_algorithm(
    algorithm: str,
    with_noise: bool = True
) -> List[Tuple[str, str]]:
    """
    Get recommended device configurations for an algorithm.
    
    Returns list of (device_name, mode) tuples.
    """
    # LRET modes to test
    lret_configs = [
        ('qlret.mixed', 'sequential'),
        ('qlret.mixed', 'batched'),
        ('qlret.mixed', 'parallel'),
        ('qlret.mixed', 'openmp'),
    ]
    
    # Comparison devices
    if with_noise:
        comparison_configs = [
            ('default.mixed', 'default'),
        ]
    else:
        comparison_configs = [
            ('default.mixed', 'default'),
            ('default.qubit', 'default'),
            ('lightning.qubit', 'default'),
        ]
    
    return lret_configs + comparison_configs


def check_device_availability() -> Dict[str, bool]:
    """Check which devices are available in the current environment."""
    available = {}
    
    # Check LRET
    try:
        from qlret import QLRETDevice
        dev = qml.device('qlret.mixed', wires=2)
        available['qlret.mixed'] = True
    except:
        available['qlret.mixed'] = False
    
    # Check comparison devices
    for name in COMPARISON_DEVICES:
        try:
            dev = qml.device(name, wires=2)
            available[name] = True
        except:
            available[name] = False
    
    return available


def print_device_availability():
    """Print device availability status."""
    avail = check_device_availability()
    
    print("\nDevice Availability:")
    print("-" * 40)
    for name, is_avail in avail.items():
        status = "✅ Available" if is_avail else "❌ Not available"
        print(f"  {name:<20} {status}")
    print("-" * 40)


if __name__ == "__main__":
    print_device_availability()
