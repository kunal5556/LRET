"""
GHZ State Creation

This example creates a Greenberger-Horne-Zeilinger (GHZ) state, which is a
maximally entangled state of three or more qubits.

GHZ state: |GHZ⟩ = (|000...0⟩ + |111...1⟩) / √2
"""

from qlret import QuantumSimulator
import numpy as np

def create_ghz_state(n_qubits):
    """
    Create a GHZ state on n qubits.
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        QuantumSimulator with GHZ state prepared
    """
    sim = QuantumSimulator(n_qubits=n_qubits)
    
    # Apply Hadamard to first qubit
    sim.h(0)
    
    # Apply CNOT from qubit 0 to all other qubits
    for i in range(1, n_qubits):
        sim.cnot(0, i)
    
    return sim


def main():
    # Test with different numbers of qubits
    for n in [3, 4, 5]:
        print(f"\n{'='*50}")
        print(f"GHZ State with {n} qubits")
        print(f"{'='*50}")
        
        sim = create_ghz_state(n)
        
        # Measure
        shots = 10000
        results = sim.measure_all(shots=shots)
        
        # Expected outcomes
        all_zeros = '0' * n
        all_ones = '1' * n
        
        print(f"\nMeasurement results ({shots} shots):")
        for outcome in [all_zeros, all_ones]:
            count = results.get(outcome, 0)
            prob = count / shots
            print(f"  |{outcome}⟩: {count:5d} ({prob:.4f})")
        
        # Check for unwanted outcomes
        other_outcomes = sum(count for outcome, count in results.items() 
                           if outcome not in [all_zeros, all_ones])
        
        if other_outcomes == 0:
            print(f"\n✓ Perfect GHZ state!")
        else:
            print(f"\n⚠ Observed {other_outcomes} measurements in unexpected states")
        
        # Verify state properties
        print(f"\nState properties:")
        print(f"  Rank: {sim.current_rank}")
        print(f"  Pure state: {sim.current_rank == 1}")
        
        # Compute reduced density matrix of first qubit
        rho_0 = sim.get_reduced_density_matrix([0])
        purity = np.trace(rho_0 @ rho_0).real
        print(f"  Reduced state purity (qubit 0): {purity:.4f}")
        print(f"  (Pure: 1.0, Maximally mixed: 0.5)")


if __name__ == "__main__":
    main()
