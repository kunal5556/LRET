"""
Grover's Search Algorithm Example

This example demonstrates Grover's algorithm for unstructured search,
which provides a quadratic speedup over classical search algorithms.
"""

import numpy as np
from qlret import QuantumSimulator
import matplotlib.pyplot as plt
from typing import List, Callable


def grover_oracle(sim, target, qubits):
    """
    Oracle that marks the target state by flipping its phase.
    
    Args:
        sim: QuantumSimulator instance
        target: Target state as integer (e.g., 3 for |011⟩)
        qubits: List of qubit indices
    """
    n = len(qubits)
    
    # Convert target to binary
    binary = format(target, f'0{n}b')
    
    # Apply X gates to qubits that should be 0 in target
    for i, bit in enumerate(binary):
        if bit == '0':
            sim.x(qubits[i])
    
    # Multi-controlled Z gate
    if n == 2:
        sim.cz(qubits[0], qubits[1])
    elif n == 3:
        # Use ancilla or decompose into 2-qubit gates
        # For simplicity, use multi-controlled Z
        # CZ on all qubits
        sim.cz(qubits[0], qubits[1])
        sim.cz(qubits[1], qubits[2])
        sim.cz(qubits[0], qubits[2])
    else:
        # General multi-controlled Z (simplified)
        # In practice, decompose into 2-qubit gates
        for i in range(n-1):
            sim.cz(qubits[i], qubits[i+1])
    
    # Undo X gates
    for i, bit in enumerate(binary):
        if bit == '0':
            sim.x(qubits[i])


def grover_diffusion(sim, qubits):
    """
    Grover diffusion operator (inversion about average).
    
    Args:
        sim: QuantumSimulator instance
        qubits: List of qubit indices
    """
    n = len(qubits)
    
    # Apply H to all qubits
    for q in qubits:
        sim.h(q)
    
    # Apply X to all qubits
    for q in qubits:
        sim.x(q)
    
    # Multi-controlled Z
    if n == 2:
        sim.cz(qubits[0], qubits[1])
    elif n == 3:
        sim.cz(qubits[0], qubits[1])
        sim.cz(qubits[1], qubits[2])
        sim.cz(qubits[0], qubits[2])
    else:
        for i in range(n-1):
            sim.cz(qubits[i], qubits[i+1])
    
    # Apply X to all qubits
    for q in qubits:
        sim.x(q)
    
    # Apply H to all qubits
    for q in qubits:
        sim.h(q)


def grover_search(n_qubits, target, n_iterations=None):
    """
    Run Grover's algorithm.
    
    Args:
        n_qubits: Number of qubits (search space size = 2^n_qubits)
        target: Target state to search for
        n_iterations: Number of Grover iterations (optimal if None)
    
    Returns:
        QuantumSimulator: Simulator after running Grover's algorithm
    """
    sim = QuantumSimulator(n_qubits)
    qubits = list(range(n_qubits))
    
    # Initialize to uniform superposition
    for q in qubits:
        sim.h(q)
    
    # Optimal number of iterations
    if n_iterations is None:
        N = 2**n_qubits
        n_iterations = int(np.pi * np.sqrt(N) / 4)
    
    # Grover iterations
    for _ in range(n_iterations):
        grover_oracle(sim, target, qubits)
        grover_diffusion(sim, qubits)
    
    return sim


def basic_grover_example():
    """Basic Grover search for 2 qubits."""
    print("="*60)
    print("Basic Grover's Algorithm (2 qubits)")
    print("="*60)
    
    n_qubits = 2
    N = 2**n_qubits
    target = 3  # Search for |11⟩
    
    print(f"\nSearch space size: {N}")
    print(f"Target state: |{format(target, f'0{n_qubits}b')}⟩")
    
    # Run Grover's algorithm
    sim = grover_search(n_qubits, target)
    
    # Measure multiple times
    n_shots = 1000
    counts = {}
    
    for _ in range(n_shots):
        # Reset to state before measurement
        sim_copy = grover_search(n_qubits, target)
        
        # Measure all qubits
        results = [sim_copy.measure_single(q) for q in range(n_qubits)]
        outcome = int(''.join(map(str, results)), 2)
        
        counts[outcome] = counts.get(outcome, 0) + 1
    
    # Display results
    print(f"\nResults after {n_shots} measurements:")
    for outcome in range(N):
        count = counts.get(outcome, 0)
        prob = count / n_shots
        binary = format(outcome, f'0{n_qubits}b')
        marker = " ← Target" if outcome == target else ""
        print(f"  |{binary}⟩: {count:4d} ({prob:.3f}){marker}")
    
    # Calculate success probability
    success_prob = counts.get(target, 0) / n_shots
    classical_prob = 1 / N
    
    print(f"\nSuccess probability:")
    print(f"  Grover: {success_prob:.3f}")
    print(f"  Classical (random): {classical_prob:.3f}")
    print(f"  Speedup factor: {success_prob/classical_prob:.2f}x")


def grover_iteration_analysis():
    """Analyze success probability vs number of iterations."""
    print("\n" + "="*60)
    print("Grover Iteration Analysis (3 qubits)")
    print("="*60)
    
    n_qubits = 3
    target = 5  # Search for |101⟩
    N = 2**n_qubits
    
    print(f"\nSearch space: {N} states")
    print(f"Target: |{format(target, f'0{n_qubits}b')}⟩")
    
    # Test different numbers of iterations
    max_iter = 10
    iteration_range = range(1, max_iter + 1)
    success_probs = []
    
    for n_iter in iteration_range:
        # Run many trials
        n_shots = 100
        success_count = 0
        
        for _ in range(n_shots):
            sim = grover_search(n_qubits, target, n_iterations=n_iter)
            
            # Measure
            results = [sim.measure_single(q) for q in range(n_qubits)]
            outcome = int(''.join(map(str, results)), 2)
            
            if outcome == target:
                success_count += 1
        
        success_prob = success_count / n_shots
        success_probs.append(success_prob)
        
        optimal_marker = " (optimal)" if n_iter == int(np.pi * np.sqrt(N) / 4) else ""
        print(f"  {n_iter:2d} iterations: {success_prob:.3f}{optimal_marker}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_range, success_probs, 'bo-', linewidth=2, markersize=8)
    
    # Mark optimal
    optimal_iter = int(np.pi * np.sqrt(N) / 4)
    plt.axvline(x=optimal_iter, color='r', linestyle='--', 
                label=f'Optimal iterations: {optimal_iter}')
    
    # Classical baseline
    plt.axhline(y=1/N, color='g', linestyle='--', 
                label=f'Classical random: {1/N:.3f}')
    
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title("Grover's Algorithm: Iterations vs Success Probability", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('grover_iterations.png', dpi=150)
    print("\nPlot saved to: grover_iterations.png")


def scaling_analysis():
    """Analyze how Grover scales with problem size."""
    print("\n" + "="*60)
    print("Grover Scaling Analysis")
    print("="*60)
    
    qubit_range = range(2, 7)  # 2 to 6 qubits
    optimal_iterations = []
    success_probs = []
    
    for n_qubits in qubit_range:
        N = 2**n_qubits
        target = N // 2  # Middle element
        
        # Optimal iterations
        n_iter = int(np.pi * np.sqrt(N) / 4)
        optimal_iterations.append(n_iter)
        
        # Run algorithm
        n_shots = 50
        success_count = 0
        
        for _ in range(n_shots):
            sim = grover_search(n_qubits, target, n_iterations=n_iter)
            
            results = [sim.measure_single(q) for q in range(n_qubits)]
            outcome = int(''.join(map(str, results)), 2)
            
            if outcome == target:
                success_count += 1
        
        success_prob = success_count / n_shots
        success_probs.append(success_prob)
        
        print(f"\n{n_qubits} qubits (N = {N}):")
        print(f"  Optimal iterations: {n_iter}")
        print(f"  Success probability: {success_prob:.3f}")
        print(f"  Classical probability: {1/N:.4f}")
    
    # Plot scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Iterations vs N
    N_values = [2**n for n in qubit_range]
    ax1.plot(N_values, optimal_iterations, 'bo-', linewidth=2, markersize=8, label='Grover')
    ax1.plot(N_values, [n/2 for n in N_values], 'r--', linewidth=2, label='Classical (N/2)')
    ax1.set_xlabel('Search Space Size (N)', fontsize=12)
    ax1.set_ylabel('Number of Queries', fontsize=12)
    ax1.set_title('Query Complexity', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Success probability
    ax2.bar(qubit_range, success_probs, color='steelblue', edgecolor='black')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Classical (average)')
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Success Probability', fontsize=12)
    ax2.set_title('Success Probability (Optimal Iterations)', fontsize=14)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('grover_scaling.png', dpi=150)
    print("\nPlot saved to: grover_scaling.png")


def multiple_solutions_example():
    """Grover search with multiple solutions."""
    print("\n" + "="*60)
    print("Grover with Multiple Solutions")
    print("="*60)
    
    n_qubits = 3
    N = 2**n_qubits
    
    # Multiple targets
    targets = [2, 5, 7]  # |010⟩, |101⟩, |111⟩
    M = len(targets)
    
    print(f"\nSearch space: {N} states")
    print(f"Number of solutions: {M}")
    print(f"Targets: {[format(t, f'0{n_qubits}b') for t in targets]}")
    
    def multi_target_oracle(sim, qubits):
        """Oracle for multiple targets."""
        for target in targets:
            grover_oracle(sim, target, qubits)
    
    # Optimal iterations for M solutions
    n_iter = int(np.pi * np.sqrt(N / M) / 4)
    print(f"\nOptimal iterations: {n_iter}")
    
    # Run algorithm with custom oracle
    sim = QuantumSimulator(n_qubits)
    qubits = list(range(n_qubits))
    
    # Initialize
    for q in qubits:
        sim.h(q)
    
    # Grover iterations with multi-target oracle
    for _ in range(n_iter):
        multi_target_oracle(sim, qubits)
        grover_diffusion(sim, qubits)
    
    # Measure multiple times
    n_shots = 1000
    counts = {}
    
    for _ in range(n_shots):
        sim_copy = QuantumSimulator(n_qubits)
        for q in qubits:
            sim_copy.h(q)
        for _ in range(n_iter):
            multi_target_oracle(sim_copy, qubits)
            grover_diffusion(sim_copy, qubits)
        
        results = [sim_copy.measure_single(q) for q in range(n_qubits)]
        outcome = int(''.join(map(str, results)), 2)
        counts[outcome] = counts.get(outcome, 0) + 1
    
    # Display results
    print(f"\nResults after {n_shots} measurements:")
    total_target_hits = 0
    for outcome in range(N):
        count = counts.get(outcome, 0)
        prob = count / n_shots
        binary = format(outcome, f'0{n_qubits}b')
        
        if outcome in targets:
            marker = " ← Target"
            total_target_hits += count
        else:
            marker = ""
        
        if count > 0:
            print(f"  |{binary}⟩: {count:4d} ({prob:.3f}){marker}")
    
    total_success = total_target_hits / n_shots
    classical_success = M / N
    
    print(f"\nTotal success probability:")
    print(f"  Grover: {total_success:.3f}")
    print(f"  Classical: {classical_success:.3f}")
    print(f"  Speedup: {total_success/classical_success:.2f}x")


def visualize_grover_circuit():
    """Display circuit diagram for Grover's algorithm."""
    print("\n" + "="*60)
    print("Grover's Algorithm Circuit")
    print("="*60)
    
    circuit = """
    Grover's Algorithm for 3 qubits:
    
    |0⟩ ──H──╭───────╮──╭──────────╮──── ... ────M──
             │       │  │          │
    |0⟩ ──H──┤Oracle├──┤Diffusion ├──── ... ────M──
             │       │  │          │
    |0⟩ ──H──╰───────╯──╰──────────╯──── ... ────M──
             
             ╰─────────────────────╯
                  Repeated ~√N times
    
    Oracle: Flips phase of target state(s)
    Diffusion: Inversion about average (amplitude amplification)
    
    Number of iterations ≈ π√(N/M)/4 where:
      - N = size of search space (2^n for n qubits)
      - M = number of solutions
    """
    
    print(circuit)


def main():
    # Example 1: Basic 2-qubit Grover
    basic_grover_example()
    
    # Example 2: Iteration analysis
    grover_iteration_analysis()
    
    # Example 3: Scaling analysis
    scaling_analysis()
    
    # Example 4: Multiple solutions
    multiple_solutions_example()
    
    # Example 5: Circuit visualization
    visualize_grover_circuit()
    
    print("\n" + "="*60)
    print("All Grover's algorithm examples complete!")
    print("="*60)
    print("\nKey Insights:")
    print("  • Grover provides quadratic speedup: O(√N) vs O(N)")
    print("  • Optimal iterations ≈ π√N/4 for single solution")
    print("  • Works with multiple solutions: π√(N/M)/4 iterations")
    print("  • Success probability ~1 with optimal iterations")
    print("  • 'Too many iterations decreases success probability")


if __name__ == "__main__":
    main()
