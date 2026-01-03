"""
Quantum Phase Estimation Example

This example demonstrates the Quantum Phase Estimation (QPE) algorithm,
which estimates the eigenvalue of a unitary operator. QPE is a key subroutine
in many quantum algorithms including Shor's algorithm.
"""

import numpy as np
from qlret import QuantumSimulator
import matplotlib.pyplot as plt


def qft_inverse(sim, qubits):
    """
    Apply inverse Quantum Fourier Transform.
    
    Args:
        sim: QuantumSimulator instance
        qubits: List of qubit indices (in order)
    """
    n = len(qubits)
    
    # Reverse order of qubits
    for i in range(n // 2):
        j = n - 1 - i
        # Swap qubits[i] and qubits[j] using CNOT gates
        sim.cx(qubits[i], qubits[j])
        sim.cx(qubits[j], qubits[i])
        sim.cx(qubits[i], qubits[j])
    
    # Apply inverse QFT gates
    for i in range(n):
        q = qubits[i]
        
        # Controlled rotations
        for j in range(i):
            control = qubits[j]
            angle = -np.pi / (2**(i - j))
            # Controlled phase rotation
            sim.rz(angle / 2, q)
            sim.cx(control, q)
            sim.rz(-angle / 2, q)
            sim.cx(control, q)
        
        # Hadamard
        sim.h(q)


def controlled_unitary(sim, control, target, unitary_power, phase):
    """
    Apply controlled-U^(2^power) where U|ψ⟩ = e^(2πi*phase)|ψ⟩.
    
    For this example, U is a phase gate: U = e^(2πi*phase*Z)
    
    Args:
        sim: QuantumSimulator instance
        control: Control qubit
        target: Target qubit  
        unitary_power: Power of the unitary (2^power applications)
        phase: Eigenvalue phase to estimate
    """
    # Total phase accumulated: 2^power * 2π * phase
    total_phase = (2**unitary_power) * 2 * np.pi * phase
    
    # Controlled phase rotation
    # This implements controlled-U^(2^power)
    sim.rz(total_phase / 2, target)
    sim.cx(control, target)
    sim.rz(-total_phase / 2, target)
    sim.cx(control, target)


def phase_estimation(n_precision, phase):
    """
    Run Quantum Phase Estimation algorithm.
    
    Args:
        n_precision: Number of precision qubits
        phase: True phase to estimate (between 0 and 1)
    
    Returns:
        int: Estimated phase as integer (divide by 2^n_precision to get phase)
    """
    # Total qubits: n_precision counting qubits + 1 eigenstate qubit
    n_total = n_precision + 1
    sim = QuantumSimulator(n_total)
    
    counting_qubits = list(range(n_precision))
    eigenstate_qubit = n_precision
    
    # Step 1: Prepare eigenstate |1⟩ (eigenstate of Z with eigenvalue -1)
    # For this example, |1⟩ is eigenstate of Z: Z|1⟩ = -|1⟩
    # Which means phase = 0.5 for Z gate
    # But we'll use a general phase gate
    sim.x(eigenstate_qubit)
    
    # Step 2: Initialize counting qubits to |+⟩
    for q in counting_qubits:
        sim.h(q)
    
    # Step 3: Apply controlled unitaries
    for i, control in enumerate(counting_qubits):
        power = n_precision - 1 - i
        controlled_unitary(sim, control, eigenstate_qubit, power, phase)
    
    # Step 4: Apply inverse QFT
    qft_inverse(sim, counting_qubits)
    
    # Step 5: Measure counting qubits
    results = [sim.measure_single(q) for q in counting_qubits]
    
    # Convert to integer
    measured_value = int(''.join(map(str, results)), 2)
    
    return measured_value


def basic_qpe_example():
    """Basic QPE example with known phase."""
    print("="*60)
    print("Basic Quantum Phase Estimation")
    print("="*60)
    
    n_precision = 4
    true_phase = 0.25  # True phase to estimate
    
    print(f"\nPrecision qubits: {n_precision}")
    print(f"True phase: {true_phase}")
    print(f"True phase (binary): {format(int(true_phase * 2**n_precision), f'0{n_precision}b')}")
    
    # Run QPE multiple times
    n_shots = 100
    estimates = []
    
    for _ in range(n_shots):
        measured = phase_estimation(n_precision, true_phase)
        estimated_phase = measured / (2**n_precision)
        estimates.append(estimated_phase)
    
    # Statistics
    unique_estimates = list(set(estimates))
    
    print(f"\nResults from {n_shots} runs:")
    for est in sorted(unique_estimates):
        count = estimates.count(est)
        prob = count / n_shots
        binary = format(int(est * 2**n_precision), f'0{n_precision}b')
        marker = " ← Correct" if abs(est - true_phase) < 1e-10 else ""
        print(f"  {est:.4f} (|{binary}⟩): {count:3d} ({prob:.3f}){marker}")
    
    # Calculate average estimate
    avg_estimate = np.mean(estimates)
    error = abs(avg_estimate - true_phase)
    
    print(f"\nAverage estimated phase: {avg_estimate:.6f}")
    print(f"True phase: {true_phase:.6f}")
    print(f"Error: {error:.6f}")


def precision_analysis():
    """Analyze how precision affects accuracy."""
    print("\n" + "="*60)
    print("Precision Analysis")
    print("="*60)
    
    true_phase = 0.387  # Arbitrary phase
    precision_range = range(2, 8)
    
    errors = []
    
    print(f"\nTrue phase: {true_phase:.6f}")
    
    for n_precision in precision_range:
        # Run multiple trials
        n_trials = 50
        estimated_phases = []
        
        for _ in range(n_trials):
            measured = phase_estimation(n_precision, true_phase)
            estimated_phase = measured / (2**n_precision)
            estimated_phases.append(estimated_phase)
        
        # Calculate average error
        avg_estimate = np.mean(estimated_phases)
        error = abs(avg_estimate - true_phase)
        errors.append(error)
        
        max_resolution = 1 / (2**n_precision)
        
        print(f"\n{n_precision} precision qubits:")
        print(f"  Resolution: {max_resolution:.6f}")
        print(f"  Average estimate: {avg_estimate:.6f}")
        print(f"  Error: {error:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(precision_range, errors, 'bo-', linewidth=2, markersize=8, label='Measured error')
    
    # Theoretical resolution
    theoretical = [1 / (2**n) for n in precision_range]
    plt.semilogy(precision_range, theoretical, 'r--', linewidth=2, label='Theoretical resolution')
    
    plt.xlabel('Number of Precision Qubits', fontsize=12)
    plt.ylabel('Phase Estimation Error', fontsize=12)
    plt.title('QPE Precision Analysis', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('qpe_precision.png', dpi=150)
    print("\nPlot saved to: qpe_precision.png")


def phase_kickback_demo():
    """Demonstrate phase kickback mechanism."""
    print("\n" + "="*60)
    print("Phase Kickback Demonstration")
    print("="*60)
    
    print("\nPhase kickback is the key mechanism in QPE.")
    print("When applying controlled-U to an eigenstate:")
    print("  |control⟩|eigenstate⟩ → e^(iφ)|control⟩|eigenstate⟩")
    print("The phase 'kicks back' to the control qubit.")
    
    # Simple example: controlled-Z on |1⟩
    sim = QuantumSimulator(2)
    
    # Prepare |+⟩|1⟩
    sim.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
    sim.x(1)  # |1⟩
    
    print("\nInitial state: |+⟩|1⟩ = (|0⟩ + |1⟩)/√2 ⊗ |1⟩")
    
    # Apply controlled-Z
    # Z|1⟩ = -|1⟩, so the phase -1 kicks back
    sim.cz(0, 1)
    
    print("After controlled-Z: (|0⟩ - |1⟩)/√2 ⊗ |1⟩ = |−⟩|1⟩")
    
    # Measure first qubit in X basis
    sim.h(0)
    result = sim.measure_single(0)
    
    print(f"\nMeasurement in Z basis after H: {result}")
    print("  (0 = |+⟩ before H, 1 = |−⟩ before H)")
    
    # The phase kickback changed |+⟩ to |−⟩


def multiple_phase_estimation():
    """Estimate phases of different unitaries."""
    print("\n" + "="*60)
    print("Multiple Phase Estimations")
    print("="*60)
    
    n_precision = 5
    test_phases = [0.0, 0.125, 0.25, 0.5, 0.75]
    
    print(f"\nPrecision: {n_precision} qubits")
    print(f"Resolution: 1/{2**n_precision} = {1/(2**n_precision):.5f}")
    
    results = []
    
    for true_phase in test_phases:
        # Run multiple trials
        n_trials = 20
        estimates = []
        
        for _ in range(n_trials):
            measured = phase_estimation(n_precision, true_phase)
            estimated_phase = measured / (2**n_precision)
            estimates.append(estimated_phase)
        
        avg_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        error = abs(avg_estimate - true_phase)
        
        results.append((true_phase, avg_estimate, std_estimate, error))
        
        print(f"\nTrue phase: {true_phase:.5f}")
        print(f"  Estimated: {avg_estimate:.5f} ± {std_estimate:.5f}")
        print(f"  Error: {error:.5f}")
    
    # Visualize
    true_phases = [r[0] for r in results]
    estimated_phases = [r[1] for r in results]
    errors = [r[3] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Estimated vs True
    ax1.plot(true_phases, estimated_phases, 'bo', markersize=10, label='Estimated')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect estimation')
    ax1.set_xlabel('True Phase', fontsize=12)
    ax1.set_ylabel('Estimated Phase', fontsize=12)
    ax1.set_title('Phase Estimation Accuracy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    
    # Errors
    ax2.bar(range(len(test_phases)), errors, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Test Case', fontsize=12)
    ax2.set_ylabel('Estimation Error', fontsize=12)
    ax2.set_title('Estimation Errors', fontsize=14)
    ax2.set_xticks(range(len(test_phases)))
    ax2.set_xticklabels([f'{p:.3f}' for p in test_phases])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qpe_multiple_phases.png', dpi=150)
    print("\nPlot saved to: qpe_multiple_phases.png")


def visualize_qpe_circuit():
    """Display QPE circuit diagram."""
    print("\n" + "="*60)
    print("Quantum Phase Estimation Circuit")
    print("="*60)
    
    circuit = """
    QPE Circuit (n precision qubits):
    
    |0⟩ ──H──●──────────────────────────╭───────╮──M──
             │                          │       │
    |0⟩ ──H──┼────●─────────────────────┤  QFT⁻¹│──M──
             │    │                     │       │
    |0⟩ ──H──┼────┼────●────────────────┤       ├──M──
             │    │    │                │       │
         ... │    │    │    ...         │       │
             │    │    │                ╰───────╯
    |ψ⟩ ─────U¹───U²───U⁴──── ... ──U^(2^(n-1))────────
    
    Where:
    - |ψ⟩ is an eigenstate of U: U|ψ⟩ = e^(2πiφ)|ψ⟩
    - U^k means U applied k times
    - QFT⁻¹ is the inverse Quantum Fourier Transform
    - Measurements give binary representation of φ
    
    Steps:
    1. Prepare counting qubits in |+⟩ (Hadamard gates)
    2. Prepare eigenstate |ψ⟩
    3. Apply controlled-U^(2^k) gates (phase kickback)
    4. Apply inverse QFT to counting qubits
    5. Measure counting qubits to read phase
    
    Result: φ ≈ (measured_value) / 2^n
    Precision: ±1/2^n
    """
    
    print(circuit)


def main():
    # Example 1: Basic QPE
    basic_qpe_example()
    
    # Example 2: Precision analysis
    precision_analysis()
    
    # Example 3: Phase kickback
    phase_kickback_demo()
    
    # Example 4: Multiple phases
    multiple_phase_estimation()
    
    # Example 5: Circuit visualization
    visualize_qpe_circuit()
    
    print("\n" + "="*60)
    print("All Quantum Phase Estimation examples complete!")
    print("="*60)
    print("\nKey Insights:")
    print("  • QPE estimates eigenvalues of unitary operators")
    print("  • Precision scales exponentially with qubit count")
    print("  • Key subroutine in Shor's algorithm and quantum chemistry")
    print("  • Uses phase kickback and inverse QFT")
    print("  • Error ≈ 1/2^n for n precision qubits")


if __name__ == "__main__":
    main()
