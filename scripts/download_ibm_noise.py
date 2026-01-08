#!/usr/bin/env python3
"""
Download IBM Quantum device noise models for LRET simulation.

This script fetches real device calibration data from IBM Quantum Experience
and exports the noise model in Qiskit-compatible JSON format.

Requirements:
    pip install qiskit qiskit-ibm-runtime

Usage:
    # Save IBM Quantum token (one-time setup)
    python download_ibm_noise.py --save-token YOUR_IBM_TOKEN
    
    # Download noise model for a specific device
    python download_ibm_noise.py --backend ibmq_quito --output quito_noise.json
    
    # List available backends
    python download_ibm_noise.py --list-backends
    
    # Download with verbose logging
    python download_ibm_noise.py --backend ibm_kyoto --output kyoto_noise.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeAthensV2, FakeYorktownV2
    from qiskit_aer.noise import NoiseModel
    FAKE_BACKENDS = {
        'FakeManila': FakeManilaV2,
        'FakeManilaV2': FakeManilaV2,
        'FakeAthens': FakeAthensV2, 
        'FakeAthensV2': FakeAthensV2,
        'FakeYorktown': FakeYorktownV2,
        'FakeYorktownV2': FakeYorktownV2,
    }
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    print("\nPlease install required packages:", file=sys.stderr)
    print("  pip install qiskit qiskit-ibm-runtime qiskit-aer", file=sys.stderr)
    sys.exit(1)


def save_token(token: str, overwrite: bool = False) -> None:
    """Save IBM Quantum token to local keychain."""
    try:
        QiskitRuntimeService.save_account(token=token, overwrite=overwrite)
        print(f"✓ Token saved successfully")
    except Exception as e:
        print(f"Error saving token: {e}", file=sys.stderr)
        sys.exit(1)


def list_backends(verbose: bool = False) -> None:
    """List available IBM Quantum backends."""
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print("\n=== Available IBM Quantum Backends ===")
        for backend in backends:
            status = "✓" if backend.status().operational else "✗"
            config = backend.configuration()
            n_qubits = config.n_qubits
            
            if verbose:
                print(f"\n{status} {backend.name}")
                print(f"   Qubits: {n_qubits}")
                print(f"   Pending Jobs: {backend.status().pending_jobs}")
                print(f"   Version: {config.backend_version}")
            else:
                print(f"{status} {backend.name:25s} ({n_qubits} qubits)")
                
    except Exception as e:
        print(f"Error listing backends: {e}", file=sys.stderr)
        print("\nNote: You may need to save your IBM token first:", file=sys.stderr)
        print("  python download_ibm_noise.py --save-token YOUR_TOKEN", file=sys.stderr)
        sys.exit(1)


def get_noise_model_from_backend(backend_name: str, verbose: bool = False) -> NoiseModel:
    """
    Retrieve noise model from IBM backend.
    
    Args:
        backend_name: Name of the IBM Quantum backend (e.g., 'ibmq_quito')
        verbose: Enable detailed logging
        
    Returns:
        NoiseModel object from qiskit-aer
    """
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        
        if verbose:
            print(f"Backend: {backend.name}")
            print(f"Version: {backend.configuration().backend_version}")
            print(f"Qubits: {backend.configuration().n_qubits}")
        
        # Build noise model from backend properties
        noise_model = NoiseModel.from_backend(backend)
        
        if verbose:
            print(f"Noise model basis gates: {noise_model.basis_gates}")
            print(f"Number of qubits: {noise_model.num_qubits}")
        
        return noise_model
        
    except Exception as e:
        print(f"Error fetching backend '{backend_name}': {e}", file=sys.stderr)
        print("\nAvailable options:", file=sys.stderr)
        print("  1. Check backend name with: --list-backends", file=sys.stderr)
        print("  2. Use fake backend with: --fake-backend FakeQuito", file=sys.stderr)
        sys.exit(1)


def get_fake_noise_model(fake_backend_name: str, verbose: bool = False) -> NoiseModel:
    """
    Get noise model from Qiskit fake backend (for testing without IBM account).
    
    Args:
        fake_backend_name: Name of fake backend (e.g., 'FakeManila', 'FakeAthens')
        verbose: Enable detailed logging
        
    Returns:
        NoiseModel object
    """
    try:
        # Look up fake backend in our registry
        fake_backend_cls = FAKE_BACKENDS.get(fake_backend_name)
        if fake_backend_cls is None:
            # Try common variations
            for name in [fake_backend_name, fake_backend_name + 'V2', 
                        fake_backend_name.replace('Fake', 'Fake') + 'V2']:
                if name in FAKE_BACKENDS:
                    fake_backend_cls = FAKE_BACKENDS[name]
                    break
        
        if fake_backend_cls is None:
            print(f"Error: Fake backend '{fake_backend_name}' not found", file=sys.stderr)
            print("\nAvailable fake backends:", file=sys.stderr)
            for name in sorted(FAKE_BACKENDS.keys()):
                print(f"  {name}", file=sys.stderr)
            sys.exit(1)
        
        backend = fake_backend_cls()
        
        if verbose:
            print(f"Using fake backend: {backend.name}")
            print(f"Qubits: {backend.num_qubits}")
        
        noise_model = NoiseModel.from_backend(backend)
        return noise_model
        
    except Exception as e:
        print(f"Error creating fake noise model: {e}", file=sys.stderr)
        sys.exit(1)


def noise_model_to_lret_json(noise_model: NoiseModel, backend_name: str = "unknown") -> Dict[str, Any]:
    """
    Convert Qiskit NoiseModel to LRET-compatible JSON format.
    
    Args:
        noise_model: Qiskit NoiseModel object
        backend_name: Name of the backend for metadata
        
    Returns:
        Dictionary in LRET noise model format
    """
    # Get Qiskit's native JSON representation
    qiskit_dict = noise_model.to_dict()
    
    # Determine num_qubits from noise instructions
    num_qubits = 0
    for error in qiskit_dict.get("errors", []):
        gate_qubits = error.get("gate_qubits", [])
        for qubits in gate_qubits:
            if qubits:
                num_qubits = max(num_qubits, max(qubits) + 1)
    
    # Add LRET-specific metadata
    lret_dict = {
        "device_name": backend_name,
        "noise_model_version": "1.0",
        "format": "qiskit_aer",
        "basis_gates": list(noise_model.basis_gates),
        "num_qubits": num_qubits,
        "errors": qiskit_dict.get("errors", []),
    }
    
    # Add backend version if available
    if "backend_version" in qiskit_dict:
        lret_dict["backend_version"] = qiskit_dict["backend_version"]
    
    return lret_dict


def download_and_save_noise_model(
    backend_name: str,
    output_path: Path,
    fake: bool = False,
    verbose: bool = False
) -> None:
    """
    Download noise model and save to JSON file.
    
    Args:
        backend_name: Name of IBM backend or fake backend
        output_path: Output JSON file path
        fake: Use fake backend instead of real IBM backend
        verbose: Enable detailed logging
    """
    print(f"Fetching noise model for: {backend_name}")
    
    # Get noise model
    if fake:
        noise_model = get_fake_noise_model(backend_name, verbose)
    else:
        noise_model = get_noise_model_from_backend(backend_name, verbose)
    
    # Convert to LRET format
    lret_json = noise_model_to_lret_json(noise_model, backend_name)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(lret_json, f, indent=2)
    
    print(f"✓ Noise model saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Device: {backend_name}")
    print(f"  Qubits: {lret_json['num_qubits']}")
    print(f"  Basis gates: {', '.join(list(noise_model.basis_gates)[:5])}" + 
          (f", ..." if len(noise_model.basis_gates) > 5 else ""))
    print(f"  Total errors: {len(lret_json['errors'])}")
    
    if verbose:
        # Count error types
        error_types = {}
        for error in lret_json['errors']:
            error_type = error.get('type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_types.items()):
            print(f"  {error_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Download IBM Quantum noise models for LRET simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Token management
    parser.add_argument('--save-token', metavar='TOKEN', help='Save IBM Quantum API token')
    parser.add_argument('--overwrite-token', action='store_true', help='Overwrite existing token')
    
    # Backend selection
    parser.add_argument('--backend', '-b', help='IBM backend name (e.g., ibmq_quito, ibm_kyoto)')
    parser.add_argument('--fake-backend', '-f', help='Use fake backend for testing (e.g., FakeQuito)')
    parser.add_argument('--list-backends', '-l', action='store_true', help='List available backends')
    
    # Output
    parser.add_argument('--output', '-o', type=Path, help='Output JSON file path')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Handle token saving
    if args.save_token:
        save_token(args.save_token, args.overwrite_token)
        return
    
    # List backends
    if args.list_backends:
        list_backends(args.verbose)
        return
    
    # Download noise model
    if args.backend or args.fake_backend:
        if not args.output:
            print("Error: --output is required when downloading noise model", file=sys.stderr)
            sys.exit(1)
        
        backend_name = args.backend if args.backend else args.fake_backend
        download_and_save_noise_model(
            backend_name,
            args.output,
            fake=bool(args.fake_backend),
            verbose=args.verbose
        )
        return
    
    # No action specified
    parser.print_help()


if __name__ == '__main__':
    main()
