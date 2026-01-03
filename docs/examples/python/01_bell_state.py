"""
Bell State Creation and Measurement

This example demonstrates how to create a Bell state (maximally entangled state)
and measure it multiple times to verify the entanglement.

Bell state: |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
"""

from qlret import QuantumSimulator
import matplotlib.pyplot as plt

def main():
    # Create a 2-qubit simulator
    sim = QuantumSimulator(n_qubits=2, noise_level=0.0)
    
    print("Creating Bell state...")
    print("Initial state: |00⟩")
    
    # Apply Hadamard to first qubit
    sim.h(0)
    print("After H(0): (|00⟩ + |10⟩) / √2")
    
    # Apply CNOT with control=0, target=1
    sim.cnot(0, 1)
    print("After CNOT(0,1): (|00⟩ + |11⟩) / √2")
    
    # Measure multiple times
    shots = 10000
    results = sim.measure_all(shots=shots)
    
    print(f"\nMeasurement results ({shots} shots):")
    for outcome, count in sorted(results.items()):
        probability = count / shots
        print(f"  |{outcome}⟩: {count:5d} ({probability:.4f})")
    
    # Verify entanglement
    if '01' not in results and '10' not in results:
        print("\n✓ Perfect entanglement! Only |00⟩ and |11⟩ observed.")
    
    # Get state information
    print(f"\nState properties:")
    print(f"  Current rank: {sim.current_rank}")
    print(f"  Is pure state: {sim.current_rank == 1}")
    
    # Visualize results
    plot_results(results, shots)


def plot_results(results, shots):
    """Plot measurement results as a bar chart."""
    outcomes = sorted(results.keys())
    counts = [results[o] for o in outcomes]
    probabilities = [c / shots for c in counts]
    
    plt.figure(figsize=(8, 6))
    plt.bar(outcomes, probabilities, color='skyblue', edgecolor='black')
    plt.xlabel('Outcome', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Bell State Measurement Results', fontsize=14)
    plt.ylim(0, 0.6)
    plt.grid(axis='y', alpha=0.3)
    
    # Add theoretical line
    plt.axhline(y=0.5, color='r', linestyle='--', label='Theoretical (0.5)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bell_state_results.png', dpi=150)
    print("\nPlot saved to: bell_state_results.png")


if __name__ == "__main__":
    main()
