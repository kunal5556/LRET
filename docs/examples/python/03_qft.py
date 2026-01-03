"""
Quantum Fourier Transform (QFT)

This example implements the Quantum Fourier Transform, a key subroutine in
many quantum algorithms including Shor's algorithm and quantum phase estimation.

The QFT transforms the computational basis states according to:
|j⟩ → (1/√N) Σₖ exp(2πijk/N) |k⟩
"""

from qlret import QuantumSimulator
import numpy as np
import matplotlib.pyplot as plt


def qft(sim, n_qubits):
    """
    Apply Quantum Fourier Transform to the first n qubits.
    
    Args:
        sim: QuantumSimulator instance
        n_qubits: Number of qubits to apply QFT to
    """
    # QFT circuit
    for j in range(n_qubits):
        sim.h(j)
        for k in range(j + 1, n_qubits):
            angle = np.pi / (2 ** (k - j))
            sim.crz(k, j, angle)
    
    # Swap qubits (reverse order)
    for j in range(n_qubits // 2):
        sim.swap(j, n_qubits - j - 1)


def inverse_qft(sim, n_qubits):
    """Apply inverse QFT."""
    # Swap qubits first
    for j in range(n_qubits // 2):
        sim.swap(j, n_qubits - j - 1)
    
    # Inverse QFT circuit
    for j in range(n_qubits - 1, -1, -1):
        for k in range(n_qubits - 1, j, -1):
            angle = -np.pi / (2 ** (k - j))
            sim.crz(k, j, angle)
        sim.h(j)


def test_qft_on_basis_state(n_qubits, initial_state):
    """
    Test QFT on a computational basis state.
    
    Args:
        n_qubits: Number of qubits
        initial_state: Initial state as integer (0 to 2^n - 1)
    """
    sim = QuantumSimulator(n_qubits=n_qubits)
    
    # Prepare initial state
    binary = format(initial_state, f'0{n_qubits}b')
    for i, bit in enumerate(binary):
        if bit == '1':
            sim.x(i)
    
    print(f"\nInitial state: |{binary}⟩ (decimal: {initial_state})")
    
    # Apply QFT
    qft(sim, n_qubits)
    
    # Get probabilities
    probs = sim.get_probabilities()
    
    # Show dominant outcomes
    print(f"\nAfter QFT - Top 5 outcomes:")
    top_indices = np.argsort(probs)[-5:][::-1]
    for idx in top_indices:
        binary_out = format(idx, f'0{n_qubits}b')
        print(f"  |{binary_out}⟩: {probs[idx]:.4f}")
    
    return probs


def test_qft_inverse(n_qubits):
    """Test that QFT followed by inverse QFT returns to original state."""
    sim = QuantumSimulator(n_qubits=n_qubits)
    
    # Prepare superposition state
    for i in range(n_qubits):
        sim.h(i)
    
    initial_probs = sim.get_probabilities()
    
    # Apply QFT then inverse QFT
    qft(sim, n_qubits)
    inverse_qft(sim, n_qubits)
    
    final_probs = sim.get_probabilities()
    
    # Check if we recovered the original state
    fidelity = np.abs(np.dot(np.sqrt(initial_probs), np.sqrt(final_probs)))
    
    print(f"\n{'='*50}")
    print(f"QFT Inverse Test ({n_qubits} qubits)")
    print(f"{'='*50}")
    print(f"Fidelity after QFT→QFT⁻¹: {fidelity:.6f}")
    
    if fidelity > 0.9999:
        print("✓ Successfully recovered initial state!")
    else:
        print("⚠ State not fully recovered")


def visualize_qft_spectrum(n_qubits, initial_state):
    """Visualize the QFT output as a frequency spectrum."""
    probs = test_qft_on_basis_state(n_qubits, initial_state)
    
    plt.figure(figsize=(12, 5))
    
    # Plot probability distribution
    plt.subplot(1, 2, 1)
    plt.bar(range(len(probs)), probs, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Basis State (Decimal)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'QFT Output Distribution\nInitial: |{initial_state}⟩, N={n_qubits} qubits', 
              fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot phase (if we had phase information)
    plt.subplot(1, 2, 2)
    # Get state vector to show phases
    try:
        state_vec = test_qft_on_basis_state(n_qubits, initial_state)
        # Create new simulator for phase info
        sim2 = QuantumSimulator(n_qubits=n_qubits)
        binary = format(initial_state, f'0{n_qubits}b')
        for i, bit in enumerate(binary):
            if bit == '1':
                sim2.x(i)
        qft(sim2, n_qubits)
        
        # Show magnitude
        plt.bar(range(2**n_qubits), probs, color='coral', 
                edgecolor='black', linewidth=0.5)
        plt.xlabel('Basis State (Decimal)', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Probability Magnitudes', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
    except:
        plt.text(0.5, 0.5, 'Phase information\nnot available\nfor mixed states', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'qft_n{n_qubits}_state{initial_state}.png', dpi=150)
    print(f"\nPlot saved to: qft_n{n_qubits}_state{initial_state}.png")


def main():
    # Test 1: QFT on specific basis states
    print("="*60)
    print("Quantum Fourier Transform Examples")
    print("="*60)
    
    n_qubits = 4
    
    # Test on |0⟩
    test_qft_on_basis_state(n_qubits, 0)
    
    # Test on |1⟩
    test_qft_on_basis_state(n_qubits, 1)
    
    # Test on |8⟩ (middle state)
    test_qft_on_basis_state(n_qubits, 8)
    
    # Test 2: QFT inverse
    test_qft_inverse(3)
    test_qft_inverse(4)
    
    # Test 3: Visualize
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print("="*60)
    visualize_qft_spectrum(4, 5)


if __name__ == "__main__":
    main()
