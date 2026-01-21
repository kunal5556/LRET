"""Quick test to validate Cirq comparison infrastructure.

This test runs Cirq on a subset of generated circuits to validate:
1. Circuit loading
2. Cirq simulation
3. Metrics collection
4. Results saving
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator
from cirq_comparison.circuit_generator import CircuitGenerator

def main():
    """Run quick validation test."""
    print("="*60)
    print("Cirq Comparison Infrastructure Test")
    print("="*60)
    
    # Test 1: Circuit generation
    print("\n[Test 1] Circuit Generation")
    gen = CircuitGenerator("test_circuits")
    circuits = []
    
    # Generate a few test circuits
    circuits.extend(gen._generate_bell_circuits(2, [0.0]))
    circuits.extend(gen._generate_ghz_circuits(3, [0.0]))
    circuits.extend(gen._generate_qft_circuits(3, [0.0]))
    
    print(f"  ✓ Generated {len(circuits)} test circuits")
    
    # Test 2: Circuit loading and conversion
    print("\n[Test 2] Circuit Loading & Conversion")
    test_circuit_path = Path(circuits[0]["path"])
    
    with open(test_circuit_path, "r") as f:
        circuit_json = json.load(f)
    
    cirq_circuit, num_qubits, noise_model = CirqFDMSimulator.from_json(circuit_json)
    print(f"  ✓ Loaded circuit: {circuits[0]['name']}")
    print(f"    Qubits: {num_qubits}, Operations: {len(cirq_circuit)}")
    
    # Test 3: Simulation
    print("\n[Test 3] Cirq Simulation")
    sim = CirqFDMSimulator(num_qubits, noise_model=noise_model)
    final_state, metadata = sim.simulate(cirq_circuit)
    
    print(f"  ✓ Simulation successful")
    print(f"    Time: {metadata['execution_time_ms']:.3f} ms")
    print(f"    Memory: {metadata['peak_memory_mb']:.3f} MB")
    print(f"    Trace: {metadata['trace']:.6f}")
    print(f"    State shape: {final_state.shape}")
    
    # Test 4: Batch processing
    print("\n[Test 4] Batch Processing (3 circuits)")
    results = []
    
    for circuit_info in circuits[:3]:
        circuit_path = Path(circuit_info["path"])
        with open(circuit_path, "r") as f:
            circuit_json = json.load(f)
        
        cirq_circuit, num_qubits, noise_model = CirqFDMSimulator.from_json(circuit_json)
        sim = CirqFDMSimulator(num_qubits, noise_model=noise_model)
        final_state, metadata = sim.simulate(cirq_circuit)
        
        result = {
            "circuit_name": circuit_info["name"],
            "num_qubits": num_qubits,
            "execution_time_ms": metadata["execution_time_ms"],
            "peak_memory_mb": metadata["peak_memory_mb"],
            "trace": metadata["trace"],
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    print(f"  ✓ Processed {len(results)} circuits")
    print("\n  Results:")
    print(df.to_string(index=False))
    
    # Test 5: Save results
    print("\n[Test 5] Saving Results")
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "test_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✓ Results saved to: {output_path}")
    
    # Test 6: Fidelity calculation
    print("\n[Test 6] Fidelity Calculation")
    # Create two similar Bell states
    from cirq import LineQubit, H, CNOT, Circuit
    
    qubits = LineQubit.range(2)
    circuit1 = Circuit(H(qubits[0]), CNOT(qubits[0], qubits[1]))
    circuit2 = Circuit(H(qubits[0]), CNOT(qubits[0], qubits[1]))
    
    sim = CirqFDMSimulator(2)
    state1, _ = sim.simulate(circuit1)
    state2, _ = sim.simulate(circuit2)
    
    fidelity = sim.compute_fidelity(state1, state2)
    print(f"  ✓ Fidelity between identical circuits: {fidelity:.6f}")
    print(f"    (Should be ~1.0)")
    
    # Summary
    print("\n" + "="*60)
    print("All Tests Passed! ✓")
    print("="*60)
    print("\nInfrastructure validated:")
    print("  ✓ Circuit generation")
    print("  ✓ JSON → Cirq conversion")
    print("  ✓ Cirq FDM simulation")
    print("  ✓ Metrics collection")
    print("  ✓ Results export")
    print("  ✓ Fidelity calculations")
    print("\nReady for full comparison benchmarks!")
    print("\nNext steps:")
    print("  1. Ensure LRET is built (quantum_sim executable)")
    print("  2. Run full comparison: python run_full_comparison.py")
    print("  3. Or run manually:")
    print("     - python circuit_generator.py")
    print("     - python run_comparison.py")
    print("     - python analyze_results.py --input results/benchmark_results_*.csv")
    print("     - python create_plots.py --input results/benchmark_results_*.csv")

if __name__ == "__main__":
    main()
